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

torch::Tensor crop_interpolate_tensor(
        const torch::Tensor &images,
        const torch::Tensor &transforms,
        const torch::Tensor &dims) {
    long W = dims[0].item().toLong();
    long H = dims[1].item().toLong();

    // Cropped output images
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(images.device());
    auto cropped_images = torch::zeros({images.size(0), images.size(1), W, H}, options);

    for (long i = 0; i < images.size(0); i++) {
        const torch::Tensor &image = images[i];
        const torch::Tensor &transform = transforms[i];

        // Get current image dimensions
        auto channels = image.size(0);
        auto w = image.size(1);
        auto h = image.size(2);

        auto avx = float(w - W);  // Available space X
        auto avy = float(h - H);  // Available space Y

        // Get the selected box starting point as subset of the image.
        // We now scale the   0,1   range to    0,(size-64)  so we cannot have invalid percentages.
        auto x1 = transform[0] * avx;
        auto y1 = transform[1] * avy;

        // We scale the image to be 64 for zoom=0, and image_size for zoom=1
        // Clip max size based on available remaining pixels
        auto x_size = torch::min(w - x1, (transform[2] * avx) + W);
        auto y_size = torch::min(h - y1, (transform[3] * avy) + H);

        for (long x = 0; x < W; x++) {
            for (long y = 0; y < H; y++) {
                auto sx = (x1 + (x_size / W) * x).toType(torch::kInt32);
                auto sy = (y1 + (y_size / H) * y).toType(torch::kInt32);;

                for (long c = 0; c < channels; c++) {
                    cropped_images[i][0][x][y] = image[0][sx][sy];
                }
            }
        }
    }

    return cropped_images;
}

torch::Tensor crop_interpolate_default(
        const torch::Tensor &images,
        const torch::Tensor &transforms,
        const torch::Tensor &dims) {
    long cW = dims[0].item().toLong();
    long cH = dims[1].item().toLong();

    // Cropped output images
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(images.device());
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
                    cropped_images[i][0][x][y] = image[0][sx][sy];
                }
            }
        }
    }
    return cropped_images;
}

torch::Tensor crop_interpolate_parallel(
        const torch::Tensor &images,
        const torch::Tensor &transforms,
        const torch::Tensor &dims) {
    long cW = dims[0].item().toLong();
    long cH = dims[1].item().toLong();

    // Cropped output images
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(images.device());
    auto cropped_images = torch::zeros({images.size(0), images.size(1), cW, cH}, options);

    torch::parallel_for(0, images.size(0), 1, [&](size_t chunk_start, size_t chunk_end) {
        for (long i = chunk_start; i < chunk_end; i++) {
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

            for (long y = 0; y < cH; y++) {
                for (long x = 0; x < cW; x++) {
                    int sx = int(x1 + (x_size / float(cW)) * float(x));
                    int sy = int(y1 + (y_size / float(cH)) * float(y));

                    for (long c = 0; c < channels; c++) {
                        cropped_images[i][c][x][y] = image[c][sx][sy];
                    }
                }
            }
        }
    });

    return cropped_images;
}


int main() {
    int B = 64;
    int C = 1;
    auto dims = torch::full({2}, 8);

    int n_devices = 2;
    int n_functions = 4;

    initialize();

    torch::Tensor base_result;
    torch::Tensor result;
    for (uint i = 7; i < (n_devices * n_functions); i++) {
        torch::Device device = torch::Device(torch::kCPU);;
        uint dev = i % n_devices;
        if (dev == 0) {
            device = torch::Device(torch::kCPU);
        } else {
            device = torch::Device(torch::kCUDA, 0);
        }


        // Input images
        torch::Tensor images = generate_dummy_images(B, C, device);
        torch::Tensor transforms = generate_dummy_transforms(B, device);
        if (i == 0) {
            base_result = crop_interpolate_tensor(images, transforms, dims);
        }

        using nano = std::chrono::nanoseconds;
        auto start = std::chrono::high_resolution_clock::now();

        uint func = i / n_devices;
        if (func == 0) {
            result = crop_interpolate_tensor(images, transforms, dims);
        } else if (func == 1) {
            result = crop_interpolate_default(images, transforms, dims);
        } else if (func == 2) {
            result = crop_interpolate_parallel(images, transforms, dims);
        } else if (func == 3 and dev == 1) {
            result = call_ci_kernel(images, transforms, dims);
        }
        auto finish = std::chrono::high_resolution_clock::now();
        std::cout << "crop_interpolate_" << func << " (" << device << ") took "
                  << std::chrono::duration_cast<nano>(finish - start).count() / 1.e6
                  << " ms\n";

        std::cout << "reference == kernel: " << torch::equal(result, base_result);
    }


    return 0;
}
