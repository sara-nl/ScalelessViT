import itertools
from typing import List
import torch
import numpy as np
import matplotlib

from model import ScalelessViT

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Resnet50:
    def __init__(self, dims, output_size):
        self.dims = dims
        self.output_size = output_size

    def __call__(self, *args, **kwargs):
        return np.zeros(1)


class Transformer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, *args, **kwargs):
        return np.zeros(1)


class Linear:
    def __init__(self, i, o):
        ...

    def __call__(self, *args, **kwargs):
        return np.random.random(4)


def main():
    calibration_iters = 10  # How many subsections of the image will we look at

    n_classes = 3  # e.g., cancer1, cancer2, nothing
    input_dims = (64, 64)
    latent_size = 32
    transformer_output_size = 32

    image_model = Resnet50(input_dims, output_size=latent_size)
    transformer_model = Transformer(latent_size, transformer_output_size)
    classifier_head = Linear(transformer_output_size, n_classes)
    transformation_head = Linear(transformer_output_size, 4)

    linear = np.arange(412 * 122) / (412 * 122)
    dataset: List[np.ndarray] = [(linear.reshape((1, 1, 412, 122)), np.random.random(1))]

    image = torch.from_numpy(dataset[0][0])
    small = torch.nn.functional.interpolate(image[:, :, 200:, 50:], [100, 50], mode="nearest")
    print(image.shape)
    print(small.shape)

    # plt.imshow(image[0][0], vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(small[0][0], vmin=0, vmax=1)
    # plt.show()

    offset_percentage = [0, 0]  # Offset range = [0, 1)
    step_scale = [1, 1]  # Step scale   = [0, 1)
    for raw_image, targets in dataset:
        latent_history = []

        w, h = raw_image.shape[2:]

        # Store previous iteration classification to compute loss on network scale_output
        previous_classes = np.zeros(1)
        for i in range(calibration_iters):
            # Get the selected box starting point as subset of the image.
            # We now scale the   0,1   range to    0,(size-64)  so we cannot have invalid percentages.
            x1 = offset_percentage[0] * (w - input_dims[0])
            y1 = offset_percentage[1] * (h - input_dims[1])

            # We scale the image to be 64 for zoom=0, and image_size for zoom=1
            x_size = (step_scale[0] * (w - input_dims[0])) + input_dims[0]
            y_size = (step_scale[1] * (h - input_dims[1])) + input_dims[1]

            # Clip max size based on available remaining pixels
            x_size = min(w - x1, x_size)
            y_size = min(h - y1, y_size)

            # Scale step to valid values for numpy array
            x_interval = (np.arange(0, 64) / 64 * x_size).astype(int)
            y_interval = (np.arange(0, 64) / 64 * y_size).astype(int)

            print(
                f"{round(x1)}-{round(x1 + x_size)}, {round(y1)}-{round(y1 + y_size)} ({round(x_size)},{round(y_size)})")

            # Make combination indices and extract sub-image
            indices = itertools.product(x_interval, y_interval)
            image = raw_image[list(indices)]

            # Get the latent from the image model
            latent = image_model(image)

            # Chain all processed latents to feed to the transformer
            latent_history.append(latent)

            # Process transformer model and receive latent across multiple image segments
            transformer_latent = transformer_model(latent_history)

            # Get classification for current image segment
            classification = classifier_head(transformer_latent)
            # Get best next offset and step scale for next subset iteration
            (
                offset_percentage[0],
                offset_percentage[1],
                step_scale[0],
                step_scale[1]
            ) = transformation_head(transformer_latent)

            # Calculate loss on the classification
            cl_weight = 1  # Class loss weight
            sl_weight = 0.2  # Scale loss weight, use to balance loss combination
            class_loss = loss_fn(classification, targets) * cl_weight

            # Scale loss subtracts previous prediction from current.
            # If previous prediction on target class was better, it should have higher loss
            # If current prediction is better, loss should be lower
            scale_loss = loss_fn(classification - previous_classes, targets) * sl_weight

            # Handle loss (maybe aggregate this over all subsegments)
            loss = class_loss + scale_loss
            # loss.backward()

            # Store previous classes to get loss next iteration
            previous_classes = classification


def loss_fn(*args, **kwargs):
    return 0


def main2():
    batch_size = 1
    model = ScalelessViT(n_classes=3)

    calibration_iters = 10  # How many subsections of the image will we look at

    dataset: List[torch.Tensor] = [(torch.rand((1, 1, 412, 122)), torch.rand(1))]

    # Initial transformation is 0, 0 (start in left upper corner) and 1, 1 (use the entire image size)
    initial_transformation = torch.stack([torch.IntTensor([0, 0, 1, 1]) for i in range(batch_size)])

    history = []
    for x, targets in dataset:
        # Store previous iteration classification to compute loss on network scale_output
        previous_classes = []
        transformation = initial_transformation
        for i in range(calibration_iters):
            print(x.shape, transformation.shape, len(history))
            classification, transformation = model(x, transformation, history)

            print(classification, transformation)

            # Calculate loss on the classification
            cl_weight = 1  # Class loss weight
            sl_weight = 0.2  # Scale loss weight, use to balance loss combination
            class_loss = model.loss(classification, targets) * cl_weight

            # Scale loss subtracts previous prediction from current.
            # If previous prediction on target class was better, it should have higher loss
            # If current prediction is better, loss should be lower
            scale_loss = model.loss(classification - previous_classes, targets) * sl_weight

            # Handle loss (maybe aggregate this over all subsegments)
            loss = class_loss + scale_loss
            print(loss)
            # loss.backward()

            # Store previous classes to get loss next iteration
            previous_classes = classification
            exit(0)


if __name__ == "__main__":
    # main()
    main2()
