# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: setup.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-11-11 18:48:26
# @Last Modified By: valarLip
# @Last Modified At: 2024-11-11 20:26:26
# @Description: This is description.

import warnings
import os
import shutil

from setuptools import setup, find_packages


import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    ROCM_HOME,
    IS_HIP_EXTENSION,
)


ck_dir = os.environ.get("CK_DIR", "/mnt/raid0/ljin1/dk/composable_kernel")
this_dir = os.path.dirname(os.path.abspath(__file__))
bd_dir = f"{this_dir}/build"
blob_dir = f"{this_dir}/blob"
PACKAGE_NAME = 'aterKernels'
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_ROCM = True
    else:
        IS_ROCM = False
else:
    if BUILD_TARGET == "cuda":
        IS_ROCM = False
    elif BUILD_TARGET == "rocm":
        IS_ROCM = True

FORCE_CXX11_ABI = False


def rename_cpp_to_cu(pths):
    ret = []
    dst = bd_dir
    for pth in pths:
        if not os.path.exists(pth):
            continue
        for entry in os.listdir(pth):
            if os.path.isdir(f'{pth}/{entry}'):
                continue
            newName = entry
            if entry.endswith(".cpp") or entry.endswith(".cu"):
                newName = entry.replace(".cpp", ".cu")
                ret.append(f'{dst}/{newName}')
            shutil.copy(f'{pth}/{entry}', f'{dst}/{newName}')
    return ret


def validate_and_update_archs(archs):
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a",
                     "gfx940", "gfx941", "gfx942", "gfx1100"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported"


ext_modules = []

if IS_ROCM:
    # use codegen get code dispatch
    if not os.path.exists(bd_dir):
        os.makedirs(bd_dir)
    if not os.path.exists(blob_dir):
        os.makedirs(blob_dir)

    print(f"\n\ntorch.__version__  = {torch.__version__}\n\n")
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
    # See https://github.com/pytorch/pytorch/pull/70650
    generator_flag = [f'-DFMOE_HSACO="{this_dir}/hsa/fmoe.co"']
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag.append("-DOLD_GENERATOR_PATH")
    if os.path.exists(ck_dir):
        generator_flag.append("-DFIND_CK")
        os.system(
            f'python {ck_dir}/example/ck_tile/02_layernorm2d/generate.py --api fwd --gen_blobs --working_path {blob_dir}')

    cc_flag = []

    archs = os.getenv("GPU_ARCHS", "native").split(";")
    validate_and_update_archs(archs)

    cc_flag = [f"--offload-arch={arch}" for arch in archs]
    cc_flag += [
        "-mllvm", "--lsr-drop-solution=1",
        "-mllvm", "-enable-post-misched=0",
        "-mllvm", "-amdgpu-coerce-illegal-types=1",
        "-mllvm", "-amdgpu-early-inline-all=true",
        "-mllvm", "-amdgpu-function-calls=false",
        "-mllvm", "--amdgpu-kernarg-preload-count=16",
    ]

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    renamed_sources = rename_cpp_to_cu([f"{this_dir}/csrc"])
    renamed_ck_srcs = rename_cpp_to_cu(
        [  # f'for other kernels'
            f"{blob_dir}",
            f"{this_dir}/csrc/impl/",
            f"{ck_dir}/example/ck_tile/12_smoothquant/instances/",
            f"{ck_dir}/example/ck_tile/13_moe_sorting/",
        ])

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"] + generator_flag,
        "nvcc":
            [
                "-O3", "-std=c++17",
                "-DUSE_PROF_API=1",
                "-D__HIP_PLATFORM_HCC__=1",
                "-D__HIP_PLATFORM_AMD__=1",
                # "-DLEGACY_HIPBLAS_DIRECT",
                "-U__HIP_NO_HALF_CONVERSIONS__",
                "-U__HIP_NO_HALF_OPERATORS__",
        ]
            + generator_flag
            + cc_flag,
    }

    include_dirs = [
        f"{this_dir}/build",
        f"{ck_dir}/include",
        f"{ck_dir}/library/include",
        f"{ck_dir}/example/ck_tile/02_layernorm2d",
        f"{ck_dir}/example/ck_tile/12_smoothquant",
        f"{ck_dir}/example/ck_tile/13_moe_sorting",
    ]

    ext_modules.append(
        CUDAExtension(
            name=PACKAGE_NAME+'_',
            sources=renamed_sources+renamed_ck_srcs,
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
        )
    )

    # ########## gradlib for tuned GEMM start here
    renamed_sources = rename_cpp_to_cu([f"{this_dir}/gradlib/csrc"])
    include_dirs = []
    ext_modules.append(
        CUDAExtension(
            name='rocsolidxgemm',
            sources=[f'{bd_dir}/rocsolgemm.cu'],
            include_dirs=include_dirs,
            # add additional libraries argument for hipblaslt
            libraries=['rocblas'],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-DLEGACY_HIPBLAS_DIRECT=ON',
                ],
                'nvcc': [
                    '-O3',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    "-ftemplate-depth=1024",
                    '-DLEGACY_HIPBLAS_DIRECT=ON',
                ] + cc_flag
            }))
    ext_modules.append(
        CUDAExtension(
            name='hipbsolidxgemm',
            sources=[f'{bd_dir}/hipbsolgemm.cu'],
            include_dirs=include_dirs,
            # add additional libraries argument for hipblaslt
            libraries=['hipblaslt'],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-DLEGACY_HIPBLAS_DIRECT=ON',
                ],
                'nvcc': [
                    '-O3',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    "-ftemplate-depth=1024",
                    '-DLEGACY_HIPBLAS_DIRECT=ON',
                ] + cc_flag
            }))
    # ########## gradlib for tuned GEMM end here
else:
    raise NotImplementedError("Only ROCM is supported")


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # calculate the maximum allowed NUM_JOBS based on cores
        max_num_jobs_cores = max(1, os.cpu_count()*0.8)
        if int(os.environ.get("MAX_JOBS", '1')) < max_num_jobs_cores:
            import psutil
            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / \
                (1024 ** 3)  # free memory in GB
            # each JOB peak memory cost is ~8-9GB when threads = 4
            max_num_jobs_memory = int(free_memory_gb / 9)

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)
# if os.path.exists(bd_dir):
#     shutil.rmtree(bd_dir)
# if os.path.exists(blob_dir):
#     shutil.rmtree(blob_dir)
# if os.path.exists(f"./.eggs"):
#     shutil.rmtree(f"./.eggs")
# if os.path.exists(f"./{PACKAGE_NAME}.egg-info"):
#     shutil.rmtree(f"./{PACKAGE_NAME}.egg-info")
# if os.path.exists('./build'):
#     shutil.rmtree(f"./build")
