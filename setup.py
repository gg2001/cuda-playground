from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name="extension_cpp",
    ext_modules=[cpp_extension.CppExtension("extension_cpp", ["main.cpp"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
