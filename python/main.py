import argparse
import os
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import ScalelessViT
import torchvision.datasets.mnist
import torchvision
from torch.utils import data

writer = SummaryWriter()
SCALE_SIZE = 5


def train(model: ScalelessViT, train_dataloader, epoch=0, calibration_iters=1, logdir="log", create_boxplot=False,
          **kwargs):
    """
    Training loop for the Scaleless model.
    This training loop keeps track of previous iteration's predictions to use for the loss of the transformation.
    Transformation loss can only be determined from the difference in classification: (p_{i} - p_{i-1})

    We use two optimizers, one for the entire model without the transformation, and one for the transformation only.

    :param model:
    :param train_dataloader:
    :param calibration_iters: amount of subsections of the image to use for prediction
    :param logdir:
    :param kwargs:
    :return:
    """
    # Initialize optimizers
    classifier_params = set(model.parameters()) - set(model.transformation_head.parameters())
    transformation_params = set(model.parameters()) - set(model.classifier_head.parameters())

    class_optimizer = torch.optim.Adam(lr=4e-4, params=classifier_params)
    transformation_optimizer = torch.optim.Adam(lr=4e-4, params=transformation_params)

    np_transformations = []

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for batch_id, (x, targets) in bar:
        transformations = []

        targets = targets.to(model.device)
        bs = targets.shape[0]

        class_optimizer.zero_grad()
        transformation_optimizer.zero_grad()

        # History for this batch
        history = model.get_initial_history(batch_size=bs)

        # Store previous iteration classification to compute loss on network scale_output
        previous_classes = torch.ones((bs, model.n_classes), device=model.device) / model.n_classes
        transformation, previous_transformation = model.get_initial_transforms(bs)
        start_loss = 0

        grid = torchvision.utils.make_grid(x)
        writer.add_image("Source images", grid, global_step=epoch)
        for i in range(calibration_iters):
            if create_boxplot:
                transformations.append(transformation)

            # Create image patch from transformation
            images = model.extract_images_with_scales(x, transformation, model.input_dims)

            # Logging information for tensorboard
            if batch_id == 0:
                grid = torchvision.utils.make_grid(images)
                writer.add_image(f"images_refine_{i}", grid
                                 .repeat_interleave(SCALE_SIZE, axis=1)
                                 .repeat_interleave(SCALE_SIZE, axis=2)
                                 .reshape(grid.shape[0], grid.shape[1] * SCALE_SIZE, grid.shape[2] * SCALE_SIZE),
                                 global_step=epoch)

                temp = torch.zeros(x.shape) + 0.5
                h = x.shape[2]
                w = x.shape[3]
                for t_i, t in enumerate(temp):
                    x1 = int(transformation[t_i][0].item() * w)
                    x2 = max(w, int(x1 + transformation[t_i][2].item() * w))
                    y1 = int(transformation[t_i][1].item() * h)
                    y2 = max(h, int(y1 + transformation[t_i][3].item() * h))
                    t[:, x1:x2, y1:y2] = 1

                grid = torchvision.utils.make_grid(temp)
                writer.add_image(
                    f"selection_refine_{i}",
                    grid
                    .repeat_interleave(SCALE_SIZE, axis=1)
                    .repeat_interleave(SCALE_SIZE, axis=2)
                    .reshape(grid.shape[0], grid.shape[1] * SCALE_SIZE, grid.shape[2] * SCALE_SIZE),
                    global_step=epoch)

            classification, next_transformation = model(images, history)

            cls_loss, tra_loss = model.compute_loss(classes=classification,
                                                    p_classes=previous_classes,
                                                    transformation=transformation,
                                                    p_transform=previous_transformation,
                                                    targets=targets)

            # Apply classification loss to classification part of network
            cls_loss.backward()
            class_optimizer.step()
            class_optimizer.zero_grad()
            if i > 0:
                # Apply transformation loss based on previous iter's transformation.
                tra_loss.backward(inputs=previous_transformation)
                transformation_optimizer.step()
                transformation_optimizer.zero_grad()
            else:
                tra_loss = torch.zeros(1)

            # Store previous classes to get loss next iteration
            previous_transformation, transformation, previous_classes = (
                transformation,
                next_transformation,
                classification
            )

            loss = (cls_loss.item(), tra_loss.item())
            # Report loss improvement over refinement iterations
            if i == 0:
                start_loss = loss
            end_loss = loss
            bar.set_postfix_str(
                f"Loss: {round(start_loss[0], 2)},{round(start_loss[1], 2)} -> {round(end_loss[0], 2)},{round(end_loss[1], 2)}")

        if create_boxplot:
            np_transformations.extend([t.cpu().detach().numpy() for t in transformations])

    if create_boxplot:
        boxplot_data = np.vstack(np_transformations)
        plt.boxplot(boxplot_data)
        plt.savefig(fname=f"boxplot_{epoch}.png")
        plt.clf()


def test(model: ScalelessViT, test_dataloader, calibration_iters, epoch=0, **kwargs):
    losses = []
    accs = []
    losses_1s = []
    accs_1s = []

    bar = tqdm(test_dataloader)
    model.eval()
    for x, targets in bar:
        # x = x.to(model.device)
        targets = targets.to(model.device)
        bs = targets.shape[0]

        # History for this batch
        history = model.get_initial_history(bs)

        # Store previous iteration classification to compute loss on network scale_output
        previous_classes = torch.ones((bs, model.n_classes), device=model.device) / model.n_classes
        transformation, previous_transformation = model.get_initial_transforms(bs)
        start_loss = 0
        for i in range(calibration_iters):
            # Create image patch from transformation
            patches = model.extract_images_with_scales(x, transformation, model.input_dims)

            classification, next_transformation = model(patches, history)

            l1, l2 = model.compute_loss(classes=classification,
                                        p_classes=previous_classes,
                                        transformation=transformation,
                                        p_transform=previous_transformation,
                                        targets=targets)

            if i == 0:
                l2 = torch.zeros((1))
            loss = l1.item() + l2.item()

            # Store previous classes to get loss next iteration
            previous_classes = classification

            losses.append(loss)
            accs.append(torch.mean((torch.argmax(previous_classes, dim=1) == targets).float().to("cpu")))
    model.train()

    losses = np.array(losses).reshape((-1, calibration_iters))
    accs = np.array(accs).reshape((-1, calibration_iters))

    for l in range(len(losses)):
        writer.add_scalars("losses", {f"iter_{i}": losses[l, i] for i in range(calibration_iters)}, global_step=epoch)
    np.set_printoptions(precision=3)
    print(
        f"Val loss: {losses.mean(axis=0)}")
    print(f"Val acc: {accs.mean(axis=0)}")


def show_image_segments(model: ScalelessViT, inp, calibration_iters, image_name="image"):
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
        ax.add_patch(Rectangle((1, 1), 2, 6,
                               edgecolor='pink',
                               facecolor='blue',
                               fill=True,
                               lw=5))
    plt.imsave(f"{image_name}_original.png", inp.moveaxis(0, -1).numpy(), format="png", vmin=0, vmax=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--interpolate_size", type=int, default=21)
    parser.add_argument("--calibration_iters", type=int, default=8)
    parser.add_argument("--inference_filename", type=str, default=None)
    parser.add_argument("--create_boxplot", type=bool, default=False)
    parser.add_argument("--logdir", type=str, default="logs/")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Limit the training dataset to N-samples. Can be used for profiling. ")
    parser.add_argument("--dataset", type=str, default="PCAM",
                        help="Select the dataset to load, currently only PCAM and MNIST are implemented.")

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
            transform=torchvision.transforms.Compose([
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.Pad([0, 0, 84, 84])
            ]),
        )

        # Split dataset
        tr = int(len(dataset) * 0.7)
        te = len(dataset) - tr
        return *data.random_split(dataset, lengths=(tr, te)), 10


def main():
    args = parse_args()

    train_dataset, test_dataset, n_classes = get_dataset(args.dataset)

    model = ScalelessViT(
        n_classes=n_classes,
        input_dims=(args.interpolate_size, args.interpolate_size),
        history_size=args.calibration_iters,
        n_heads=args.calibration_iters,
        device=args.device,
        n_channels=train_dataset[0][0].shape[0],
    )

    # Load data and model to device
    model = model.to(args.device)

    if args.inference_filename is not None:
        print("Writing inference images to file")
        model.load_state_dict(torch.load(args.inference_filename))
        for i in range(10):
            show_image_segments(model, train_dataset[i][0], args.calibration_iters,
                                image_name=f"images/im_{i}_{train_dataset[i][1]}")
        exit(0)

    # Create dataloaders
    train_dataloader = data.dataloader.DataLoader(train_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=18,
                                                  pin_memory=True,
                                                  persistent_workers=True,
                                                  prefetch_factor=4)
    test_dataloader = data.dataloader.DataLoader(test_dataset,
                                                 batch_size=1024,
                                                 num_workers=18,
                                                 pin_memory=True,
                                                 persistent_workers=True,
                                                 prefetch_factor=4)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        train(model, train_dataloader, epoch=epoch, **vars(args))
        test(model, test_dataloader, epoch=epoch, **vars(args))

        # Store model
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/model.chkpt")
    writer.close()

    # show_image_segments(model, test_dataset[0][0], args.calibration_iters)


if __name__ == "__main__":
    main()
