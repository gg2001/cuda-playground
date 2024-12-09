#include <torch/extension.h>

torch::Tensor flash_attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention", &flash_attention_cuda, "Flash attention (CUDA)");
}
