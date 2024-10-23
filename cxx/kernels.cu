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
                                        uchar *output_images, struct dims out_dims) {
    size_t batch_id = blockIdx.x;
    size_t sub_x = threadIdx.x;
    size_t sub_y = threadIdx.y;

    float *transform = &transforms[batch_id * 4];
    size_t image_size = in_dims.w * in_dims.h * in_dims.c;
    uchar *image = &images[batch_id * image_size];

    size_t output_image_size = out_dims.w * out_dims.h * out_dims.c;
    uchar *output_image = &output_images[batch_id * output_image_size];

    auto avx = float(in_dims.w - out_dims.w);  // Available space X
    auto avy = float(in_dims.h - out_dims.h);  // Available space Y

    // Get the selected box starting point as subset of the image.
    // We now scale the   0,1   range to    0,(size-64)  so we cannot have invalid percentages.
    auto x1 = transform[0] * avx;
    auto y1 = transform[1] * avy;

    // We scale the image to be 64 for zoom=0, and image_size for zoom=1
    // Clip max size based on available remaining pixels
    auto x_size = min(in_dims.w - x1, (transform[2] * avx) + out_dims.w);
    auto y_size = min(in_dims.h - y1, (transform[3] * avy) + out_dims.h);

    // Iterate x chunks if the output image size is larger than allowed block size.
    for (uint offset_x = 0; offset_x < out_dims.w; offset_x += blockDim.x) {
        int x = sub_x + offset_x;
        if (x >= out_dims.w) return;

        int sx = int(x1 + (x_size / out_dims.w) * x);
        for (uint offset_y = 0; offset_y < out_dims.h; offset_y += blockDim.y) {
            int y = sub_y + offset_y;
            if (y >= out_dims.h) return;

            int sy = int(y1 + (y_size / out_dims.h) * y);
            for (long c = 0; c < out_dims.c; c++) {
                output_image[(c * out_dims.w + y) * out_dims.h + x] = image[(c * in_dims.w + sy) * in_dims.h + sx];
            }
        }
    }
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
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(images.device());
    auto output_images = torch::zeros(out_dims_torch, options);

    // Wait for memory to finish copying
    cudaDeviceSynchronize();

    /*
     * Process kernel
     */
    dim3 blocks;
    if (output_dims.w * output_dims.h < 1024) {
        blocks = {uint(output_dims.w), uint(output_dims.h)};
    } else {
        blocks = {32, 32};
    }
    dim3 grids = {uint(input_dims.b)};

    crop_interpolate_kernel<<<grids, blocks>>>(
            images.data_ptr<uchar>(),        // Pass the pointer to the image tensor's data.
            input_dims,                      // Pass the dimensions of the input images
            transforms.data_ptr<float>(),    // Pass the transformation tensor's data
            output_images.data_ptr<uchar>(), // Output images tensor's data pointer
            output_dims                      // Output image tensor dimensions
    );
    // Wait for the CUDA device to finish processing the kernel
    gpuErrchk(cudaDeviceSynchronize());

    return output_images;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("crop_interpolate", &call_ci_kernel, "Crop Interpolate");
}
