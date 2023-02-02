#include <vector>
#include <cstdio>
#include <torch/extension.h>
#include "omp.h"
#include <ATen/ATen.h>
torch::Tensor generate_dummy_images(int B, int C) {
    int W = 28, H = 28;

    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    return torch::randint(0, 255, {B, C, W, H}, options);
}

torch::Tensor generate_dummy_transforms(int B) {
    auto options = torch::TensorOptions().dtype(torch::kFloat16);
    return torch::rand({B, 4}, options);
}


int main() {
    int B = 64;
    int C = 1;
    int cW = 8, cH = 8;

    // Cropped output images
    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    auto cropped_images = torch::zeros({B, C, cW, cH}, options);

    // Input images
    torch::Tensor images = generate_dummy_images(B, C);
    torch::Tensor transforms = generate_dummy_transforms(B);

    using milli = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();

//#pragma omp parallel for

    for (long i = 0; i < images.size(0); i++) {
        const torch::Tensor& image = images[i];
        const torch::Tensor& transform = transforms[i];

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


    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "crop_interpolate took "
              << std::chrono::duration_cast<milli>(finish - start).count()
              << " ms\n";
    return 0;
}
