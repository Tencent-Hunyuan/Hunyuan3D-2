# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the respective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from setuptools import setup, Extension
import pybind11
import sys
import platform

def get_platform_specific_args():
    system = platform.system().lower()
    cpp_std = 'c++14'  # Make configurable if needed
    
    if sys.platform == 'win32':
        compile_args = ['/O2', f'/std:{cpp_std}', '/EHsc', '/MP', '/DWIN32_LEAN_AND_MEAN', '/bigobj']
        link_args = []
        extra_includes = []
    elif system == 'linux':
        compile_args = ['-O3', f'-std={cpp_std}', '-fPIC', '-Wall', '-Wextra', '-pthread']
        link_args = ['-fPIC', '-pthread']
        extra_includes = []
    elif sys.platform == 'darwin':
        compile_args = ['-O3', f'-std={cpp_std}', '-fPIC', '-Wall', '-Wextra',
                       '-stdlib=libc++', '-mmacosx-version-min=10.14']
        link_args = ['-fPIC', '-stdlib=libc++', '-mmacosx-version-min=10.14', '-dynamiclib']
        extra_includes = []
    else:
        raise RuntimeError(f"Unsupported platform: {system}")
    
    return compile_args, link_args, extra_includes

extra_compile_args, extra_link_args, platform_includes = get_platform_specific_args()
include_dirs = [pybind11.get_include(), pybind11.get_include(user=True)]
include_dirs.extend(platform_includes)

ext_modules = [
    Extension(
        "mesh_processor",
        ["mesh_processor.cpp"],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="mesh_processor",
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0'],
    python_requires='>=3.6',
)