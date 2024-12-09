#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void flash_attention_kernel(
    const float *queries,
    const float *keys,
    const float *values,
    const int seq_length,
    const int head_dim,
    const int num_col_tiles,
    const int num_row_tiles,
    const int block_cols,
    const int block_rows,
    const float scale,
    float *running_sums,
    float *running_maxes,
    float *output)
{
  // Thread and block indices
  int thread_idx = threadIdx.x;
  int batch_idx = blockIdx.x;
  int head_idx = blockIdx.y;

  // Calculate base offsets for the current batch and attention head
  int qkv_index = (batch_idx * gridDim.y * seq_length * head_dim) + (head_idx * seq_length * head_dim);
  int stats_offset = (batch_idx * gridDim.y * seq_length) + (head_idx * seq_length);

  // Allocate shared memory for tiled computation
  extern __shared__ float shared_memory[];
  int size = block_cols * head_dim;

  // Partition shared memory for different matrices
  float *query_tile = shared_memory;           // Current query block
  float *key_tile = &query_tile[size];    // Current key block
  float *value_tile = &key_tile[size];    // Current value block
  float *attn_scores = &value_tile[size]; // Attention scores for current block

  // Process each column tile
  for (int col_tile = 0; col_tile < num_col_tiles; col_tile++)
  {
    // Step 1: Load key and value tiles into shared memory
    if (thread_idx < block_cols)
    {
      for (int dim = 0; dim < head_dim; dim++)
      {
        int global_col_idx = col_tile * block_cols + thread_idx;
        if (global_col_idx < seq_length)
        {
          key_tile[(thread_idx * head_dim) + dim] = keys[qkv_index + (global_col_idx * head_dim) + dim];
          value_tile[(thread_idx * head_dim) + dim] = values[qkv_index + (global_col_idx * head_dim) + dim];
        }
      }
    }
    __syncthreads();

    // Process each row tile
    for (int row_tile = 0; row_tile < num_row_tiles; row_tile++)
    {
      if (thread_idx < block_rows)
      {
        // Step 2: Load query tile into shared memory
        int global_row_idx = row_tile * block_rows + thread_idx;
        for (int dim = 0; dim < head_dim; dim++)
        {
          if (global_row_idx < seq_length)
          {
            query_tile[(thread_idx * head_dim) + dim] = queries[qkv_index + (global_row_idx * head_dim) + dim];
          }
        }

        // Load previous statistics for numerical stability
        float prev_max = running_maxes[stats_offset + (block_rows * row_tile) + thread_idx];
        float prev_sum = running_sums[stats_offset + (block_rows * row_tile) + thread_idx];

        // Step 3: Compute attention scores and find maximum
        float current_max = -INFINITY;
        for (int col = 0; col < block_cols; col++)
        {
          int global_col = col_tile * block_cols + col;
          int global_row = row_tile * block_rows + thread_idx;

          // Apply causal masking (upper triangular mask)
          if (global_col > global_row)
          {
            attn_scores[(block_cols * thread_idx) + col] = -INFINITY;
            continue;
          }

          // Compute dot product between query and key
          float dot_product = 0.0f;
          for (int dim = 0; dim < head_dim; dim++)
          {
            dot_product += query_tile[(thread_idx * head_dim) + dim] * key_tile[(col * head_dim) + dim];
          }

          // Scale dot product and store
          float score = dot_product * scale;
          attn_scores[(block_cols * thread_idx) + col] = score;
          current_max = max(current_max, score);
        }

        // Step 4: Compute softmax values
        float current_sum = 0.0f;
        for (int col = 0; col < block_cols; col++)
        {
          float score = attn_scores[(block_cols * thread_idx) + col];
          float exp_score = __expf(score - current_max);
          attn_scores[(block_cols * thread_idx) + col] = exp_score;
          current_sum += exp_score;
        }

        // Step 5: Update running statistics
        float new_max = max(prev_max, current_max);
        float scaled_prev_sum = __expf(prev_max - new_max) * prev_sum;
        float scaled_current_sum = __expf(current_max - new_max) * current_sum;
        float new_sum = scaled_prev_sum + scaled_current_sum;

        // Step 6: Compute weighted values and update output
        for (int dim = 0; dim < head_dim; dim++)
        {
          float weighted_sum = 0.0f;
          for (int col = 0; col < block_cols; col++)
          {
            weighted_sum += attn_scores[(block_cols * thread_idx) + col] *
                            value_tile[(col * head_dim) + dim];
          }

          if (global_row_idx < seq_length)
          {
            // Combine previous and current results with proper scaling
            output[qkv_index + (global_row_idx * head_dim) + dim] =
                (1.0f / new_sum) * ((scaled_prev_sum * output[qkv_index + (global_row_idx * head_dim) + dim]) +
                                    (scaled_current_sum * weighted_sum / current_sum));
          }
        }

        // Update running statistics in global memory
        if (global_row_idx < seq_length)
        {
          running_maxes[stats_offset + (block_rows * row_tile) + thread_idx] = new_max;
          running_sums[stats_offset + (block_rows * row_tile) + thread_idx] = new_sum;
        }
      }
    }
    __syncthreads();
  }
}

torch::Tensor flash_attention_cuda(torch::Tensor queries, torch::Tensor keys, torch::Tensor values)
{
  // Constants for tiling
  const int block_cols = 16; // Tile size for columns (keys/values)
  const int block_rows = 16; // Tile size for rows (queries)

  // Extract tensor dimensions
  const int batch_size = queries.size(0); // Batch size
  const int num_heads = queries.size(1);  // Number of attention heads
  const int seq_length = queries.size(2); // Sequence length
  const int head_dim = queries.size(3);   // Dimension of each head

  // Calculate number of tiles needed
  const int num_col_tiles = (seq_length + block_cols - 1) / block_cols;
  const int num_row_tiles = (seq_length + block_rows - 1) / block_rows;

  // Calculate attention scaling factor
  const float scale = 1.0f / sqrt(head_dim);

  // Initialize output and auxiliary tensors
  auto output = torch::zeros_like(queries);
  auto running_sums = torch::zeros({batch_size, num_heads, seq_length}, queries.options());
  auto running_maxes = torch::full({batch_size, num_heads, seq_length}, -INFINITY, queries.options());

  // Calculate shared memory requirements
  const int shared_memory_size = (3 * block_cols * head_dim + block_cols * block_rows) * sizeof(float);

  // Verify shared memory availability
  int max_shared_memory;
  cudaDeviceGetAttribute(&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  if (shared_memory_size > max_shared_memory)
  {
    throw std::runtime_error("Requested shared memory exceeds device limit");
  }

  // Configure kernel launch parameters
  dim3 grid(batch_size, num_heads);        // Grid dimensions for batch and heads
  dim3 block(max(block_cols, block_rows)); // Block dimension for maximum tile size

  // Launch kernel
  flash_attention_kernel<<<grid, block, shared_memory_size>>>(
      queries.data_ptr<float>(),
      keys.data_ptr<float>(),
      values.data_ptr<float>(),
      seq_length,
      head_dim,
      num_col_tiles,
      num_row_tiles,
      block_cols,
      block_rows,
      scale,
      running_sums.data_ptr<float>(),
      running_maxes.data_ptr<float>(),
      output.data_ptr<float>());

  return output;
}