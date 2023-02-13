//
// Created by duncan on 09-02-23.
//

#ifndef CXX_KERNELS_CUH
#define CXX_KERNELS_CUH

#include <torch/extension.h>
#include <ATen/ATen.h>

void initialize();
torch::Tensor call_ci_kernel(const torch::Tensor &images,
                             const torch::Tensor &transforms,
                             const torch::Tensor &dims);

#endif //CXX_KERNELS_CUH
