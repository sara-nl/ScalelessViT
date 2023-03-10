#include <vector>
#include <cstdio>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "kernels.cuh"

torch::Tensor generate_dummy_images(int B, int C, torch::Device dev) {
    int W = 28, H = 28;

    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    return torch::randint(0, 255, {B, C, W, H}, options);
}

torch::Tensor generate_dummy_transforms(int B, torch::Device dev) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    return torch::rand({B, 4}, options);
}


torch::Tensor crop_interpolate_default(
        const torch::Tensor &images,
        const torch::Tensor &transforms,
        const torch::Tensor &dims) {
    long cW = dims[0].item().toLong();
    long cH = dims[1].item().toLong();

    // Cropped output images
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(images.device());
    auto cropped_images = torch::zeros({images.size(0), images.size(1), cW, cH}, options);

    for (long i = 0; i < images.size(0); i++) {
        const torch::Tensor &image = images[i];
        const torch::Tensor &transform = transforms[i];

        // Get current image dimensions
        size_t channels = image.size(0);
        size_t w = image.size(1);
        size_t h = image.size(2);

        auto avx = float(w - cW);  // Available space X
        auto avy = float(h - cH);  // Available space Y

        // Get the selected box starting point as subset of the image.
        // We now scale the   0,1   range to    0,(size-64)  so we cannot have invalid percentages.
        float x1 = transform[0].item().toFloat() * avx;
        float y1 = transform[1].item().toFloat() * avy;

        // We scale the image to be 64 for zoom=0, and image_size for zoom=1
        // Clip max size based on available remaining pixels
        float x_size = std::fmin(float(w) - x1, (transform[2].item().toFloat() * avx) + float(cW));
        float y_size = std::fmin(float(h) - y1, (transform[3].item().toFloat() * avy) + float(cH));

        for (long x = 0; x < cW; x++) {
            for (long y = 0; y < cH; y++) {
                int sx = int(x1 + (x_size / float(cW)) * float(x));
                int sy = int(y1 + (y_size / float(cH)) * float(y));

                for (long c = 0; c < channels; c++) {
                    cropped_images[i][c][x][y] = image[c][sx][sy].item().toFloat() / 255;
                }
            }
        }
    }
    return cropped_images;
}

int main() {
    int B = 64;
    int C = 3;

    auto options = torch::TensorOptions().dtype(torch::kLong);
    torch::Tensor dims = torch::full({2}, 8, options);

    /*
     * Create reference images and input data
     */
    auto cpu_device = torch::Device(torch::kCPU);

    torch::Tensor images = generate_dummy_images(B, C, cpu_device);
    torch::Tensor transforms = generate_dummy_transforms(B, cpu_device);

    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor base_result = crop_interpolate_default(images, transforms, dims);
    auto finish = std::chrono::high_resolution_clock::now();

    /*
     * Output timing result and validate images are the same.
     */
    using nano = std::chrono::nanoseconds;
    std::cout << "crop_interpolate_default (" << cpu_device << ") took "
              << std::chrono::duration_cast<nano>(finish - start).count() / 1.e6
              << " ms\n";

    /*
     * Initializing the CUDA device occurs before the timing starts, as this is a one-time action.
     * Counting this as part of the kernel compute time will make it an unfair comparison compared to CPU.
     */
    torch::Device device = torch::Device(torch::kCUDA, 0);
    initialize();

    /*
     * Run kernel
     */
    auto gpu_images = images.to(device);
    auto gpu_transforms = transforms.to(device);
    auto gpu_dims = dims.to(device);

    // Input images
    start = std::chrono::high_resolution_clock::now();
    torch::Tensor result = call_ci_kernel(gpu_images, gpu_transforms, gpu_dims);
    finish = std::chrono::high_resolution_clock::now();

    std::cout << result.to(cpu_device) - base_result << std::endl;

    /*
     * Output timing result and validate images are the same.
     */
    using nano = std::chrono::nanoseconds;
    std::cout << "call_ci_kernel (" << device << ") took "
              << std::chrono::duration_cast<nano>(finish - start).count() / 1.e6
              << " ms\n";

    float threshold = 1e-6;
    std::cout << "reference == kernel : (0 = bad) " << (torch::max(result.to(cpu_device) - base_result) < threshold) << std::endl;

    return 0;
}
