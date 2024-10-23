import argparse

import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils import data

from model import SegmentSelectionModel


def show_image_segments(model: SegmentSelectionModel, inp, calibration_iters, image_name="image"):
    # History for this batch
    history = model.get_initial_history(1)

    x = inp.to(model.device)

    # Store previous iteration classification to compute loss on network scale_output
    transformation = model.get_initial_transforms(1)
    old_transformations = []
    for i in range(calibration_iters):
        old_transformations.append(transformation.detach())

        image = model.extract_images_with_scales(x.unsqueeze(0), transformation, model.input_dims)
        arr = image.squeeze(0).detach().cpu().numpy()

        classification, transformation = model(image, history)

        plt.imsave(f"{image_name}_{i}.png", np.moveaxis(arr, 0, -1), format="png", vmin=0, vmax=1)
        print(f"[{image_name}] Classification {i}:", torch.argmax(classification).item())

    fig, ax = plt.subplots()
    for transform in old_transformations:
        ax.add_patch(
            Rectangle(
                (1, 1), 2, 6,
                edgecolor='pink',
                facecolor='blue',
                fill=True,
                lw=5
            )
        )
    plt.imsave(f"{image_name}_original.png", inp.moveaxis(0, -1).numpy(), format="png", vmin=0, vmax=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_type", type=str, default="image", choices=["image", "segment"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--interpolate_size", type=int, default=21)
    parser.add_argument("--calibration_iters", type=int, default=8)
    parser.add_argument("--inference_filename", type=str, default=None)
    parser.add_argument("--create_boxplot", type=bool, default=False)
    parser.add_argument("--logdir", type=str, default="logs/")
    parser.add_argument(
        "--max_samples", type=int, default=-1,
        help="Limit the training dataset to N-samples. Can be used for profiling. "
    )
    parser.add_argument(
        "--dataset", type=str, default="PCAM",
        help="Select the dataset to load, currently only PCAM and MNIST are implemented."
    )

    return parser.parse_args()


def get_dataset(name):
    if name == "PCAM":
        train_dataset = torchvision.datasets.PCAM(
            root="datasets/",
            split="train",
            download=True,
            transform=torchvision.transforms.PILToTensor(),
        )
        test_dataset = torchvision.datasets.PCAM(
            root="datasets/",
            split="test",
            download=True,
            transform=torchvision.transforms.PILToTensor(),
        )
        return train_dataset, test_dataset, 2
    elif name == "MNIST":
        dataset = torchvision.datasets.mnist.MNIST(
            root="datasets/",
            download=True,
            transform=torchvision.transforms.PILToTensor(),
        )
        # Split dataset
        tr = int(len(dataset) * 0.7)
        te = len(dataset) - tr
        return *data.random_split(dataset, lengths=(tr, te)), 10
    elif name == "MNIST_TRANSFORM":
        dataset = torchvision.datasets.mnist.MNIST(
            root="datasets/",
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.PILToTensor(),
                    torchvision.transforms.Pad([0, 0, 84, 84])
                ]
            ),
        )

        # Split dataset
        tr = int(len(dataset) * 0.7)
        te = len(dataset) - tr
        return *data.random_split(dataset, lengths=(tr, te)), 10
