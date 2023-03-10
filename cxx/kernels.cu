//
// Created by duncan on 09-02-23.
//

#include "kernels.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define uchar unsigned char

struct dims {
    long b;
    long c;
    long w;
    long h;
};

/**
 * This kernel performs a chained crop-interpolate operation. This means the batch of input images will be cropped to
 * a certain size, and then interpolated with a closest pixel algorithm.
 * The input image array is assumed to be of format [B, C, X, Y] in unsigned characters.
 * The output image array will be [B, C, x, y] in floats.
 * The output image has scaled the input values from [0, 255] to [0, 1]
 * It supports any amount of channels, but the output x, y is limited to the maximum thread-block size on the device.
 */
__global__ void crop_interpolate_kernel(uchar *images, struct dims in_dims, float *transforms,
                                        float *output_images, struct dims out_dims) {
    /*
     *  This function will be called many times in parallel, and you will need to make sure no threads are interfering.
     *  Remember the GPU parallelism layout:
     *
     *     [          grid 0         ] [          grid 1         ]
     *     [ b0 ] [ b1 ] [ b2 ] [ b3 ] [ b0 ] [ b1 ] [ b2 ] [ b3 ]
     */
}

void initialize() {
    cudaFree(0);
}

torch::Tensor call_ci_kernel(const torch::Tensor &images,
                             const torch::Tensor &transforms,
                             const torch::Tensor &dims) {
    // Check if the tensors we are operating on actually are on the GPU.
    assert(images.device().is_cuda());
    assert(transforms.device().is_cuda());

    // Set current CUDA device to the one the tensors are residing.
    // This is to make sure we do not need to copy the tensor to another CUDA device to compute on them.
    cudaSetDevice(images.device().index());

    /*
     * Store in and output dimensions
     */
    struct dims input_dims = {
            images.size(0),
            images.size(1),
            images.size(2),
            images.size(3)
    };
    struct dims output_dims = {
            images.size(0),
            images.size(1),
            dims[0].item().toLong(),
            dims[1].item().toLong()
    };

    /*
     * Initialize output tensor
     */
    std::vector<long> vec = {
            long(output_dims.b), long(output_dims.c),
            long(output_dims.w), long(output_dims.h)
    };
    torch::IntArrayRef out_dims_torch(vec);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(images.device());
    auto output_images = torch::zeros(out_dims_torch, options);

    // Wait for memory to finish copying
    cudaDeviceSynchronize();

    /*
     * Process kernel
     */
    dim3 blocks = {1};
    dim3 grids = {1};
    crop_interpolate_kernel<<<grids, blocks>>>(
            images.data_ptr<uchar>(),        // Pass the pointer to the image tensor's data.
            input_dims,                      // Pass the dimensions of the input images
            transforms.data_ptr<float>(),    // Pass the transformation tensor's data
            output_images.data_ptr<float>(), // Output images tensor's data pointer
            output_dims                      // Output image tensor dimensions
    );
    // Wait for the CUDA device to finish processing the kernel
    gpuErrchk(cudaDeviceSynchronize());

    return output_images;
}

//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    m.def("crop_interpolate", &call_ci_kernel, "Crop Interpolate");
//}
