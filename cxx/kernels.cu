//
// Created by duncan on 09-02-23.
//

#include "kernels.cuh"

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
__global__ void ci_perpixel_kernel(uchar *images, struct dims in_dims, float *transforms,
                                   float *output_images, struct dims out_dims) {

    size_t batch_id = blockIdx.x;
    size_t x = threadIdx.x;
    size_t y = threadIdx.y;

    float *transform = &transforms[batch_id * 4];
    size_t image_size = in_dims.w * in_dims.h * in_dims.c;
    uchar *image = &images[batch_id * image_size];

    size_t output_image_size = out_dims.w * out_dims.h * out_dims.c;
    float *output_image = &output_images[batch_id * output_image_size];

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

    int sx = int(x1 + (x_size / out_dims.w) * x);
    int sy = int(y1 + (y_size / out_dims.h) * y);

    for (long c = 0; c < out_dims.c; c++) {
        output_image[(c * out_dims.w + x) * out_dims.h + y] = float(image[(c * in_dims.w + sx) * in_dims.h + sy]) / 255;
    }
}

void initialize() {
    cudaFree(0);
}

torch::Tensor call_ci_kernel(const torch::Tensor &images,
                             const torch::Tensor &transforms,
                             const torch::Tensor &dims) {
    // There are no checks in place for CUDA-hosted tensors for now.
    // This can easily be implemented later.
    assert(images.device().is_cuda());
    assert(transforms.device().is_cuda());

    // Set current CUDA device to the one the tensors are residing
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
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(images.device());
    auto output_images = torch::zeros(out_dims_torch, options);


    // Wait for memory to finish copying
    cudaDeviceSynchronize();

    /*
     * Process kernel
     */
    dim3 blocks = {uint(output_dims.w), uint(output_dims.h)};
    dim3 grids = {uint(input_dims.b)};
    ci_perpixel_kernel<<<grids, blocks>>>(
            images.data_ptr<uchar>(),
            input_dims,
            transforms.data_ptr<float>(),
            output_images.data_ptr<float>(),
            output_dims
    );

    cudaDeviceSynchronize();

    return output_images;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("crop_interpolate", &call_ci_kernel, "Crop Interpolate");
}
