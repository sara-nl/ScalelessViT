import argparse
import os
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from model import ScalelessViT
import torchvision.datasets.mnist
from torch.utils import data


def train(model, train_dataloader, calibration_iters):
    classifier_params = set(model.parameters()) - set(model.transformation_head.parameters())
    transformation_params = set(model.parameters()) - set(model.classifier_head.parameters()) - set(
        model.image_model.parameters())

    class_optimizer = torch.optim.Adam(lr=2e-4, params=classifier_params)
    transformation_optimizer = torch.optim.Adam(lr=2e-4, params=transformation_params)

    bar = tqdm(train_dataloader)
    for x, targets in bar:
        targets = targets.to(model.device)
        # x = x.to(model.device)
        bs = targets.shape[0]

        class_optimizer.zero_grad()
        transformation_optimizer.zero_grad()

        # History for this batch
        history = model.get_initial_history(batch_size=bs)

        # Store previous iteration classification to compute loss on network scale_output
        previous_classes = torch.ones((bs, model.n_classes), device=model.device) / model.n_classes
        transformation = model.get_initial_transform(bs)
        start_loss = 0
        end_loss = 0
        for i in range(calibration_iters):
            # Create image patch from transformation
            images = model.extract_images_with_scales(x, transformation, model.input_dims)
            classification, next_transformation = model(images, history)

            cls_loss, tra_loss = model.compute_loss(classification, previous_classes, targets=targets)

            # Apply classification loss to classification part of network
            cls_loss.backward()
            class_optimizer.step()
            class_optimizer.zero_grad()

            if i > 0:
                # Apply transformation loss based on previous iter's transformation.
                tra_loss.backward(inputs=transformation)
                transformation_optimizer.step()
                transformation_optimizer.zero_grad()

            # Store previous classes to get loss next iteration
            previous_classes = classification
            transformation = next_transformation

            loss = cls_loss.item() + tra_loss.item()
            # Report loss improvement over refinement iterations
            if i == 0:
                start_loss = loss
            end_loss = loss
            bar.set_postfix_str(f"Loss: {round(start_loss, 2)} -> {round(end_loss, 2)}")

        # model.profiler.print_stats()
        # exit(0)


def test(model, test_dataloader, calibration_iters):
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
        transformation = model.get_initial_transform(bs)
        for i in range(calibration_iters):
            images = model.extract_images_with_scales(x, transformation, model.input_dims)
            classification, transformation = model(images, history)

            l1, l2 = model.compute_loss(classification, previous_classes, targets=targets)
            loss = l1.item() + l2.item()

            # Store previous classes to get loss next iteration
            previous_classes = classification

            if i == 0:
                accs_1s.append(torch.mean((torch.argmax(previous_classes, dim=1) == targets).float()))
                losses_1s.append(loss)

            losses.append(loss)
        accs.append(torch.mean((torch.argmax(previous_classes, dim=1) == targets).float()))
    model.train()
    print(
        f"Val loss: {sum(losses[calibration_iters - 1::calibration_iters]) / (len(losses) // calibration_iters)} (1-shot: {sum(losses_1s) / len(losses_1s)})")
    print(f"Val acc: {sum(accs) / len(accs)} (1-shot: {sum(accs_1s) / len(accs_1s)}")


def show_image_segments(model: ScalelessViT, inp, calibration_iters, image_name="image"):
    # History for this batch
    history = model.get_initial_history(1)

    x = inp.to(model.device)

    print(x.shape)
    # Store previous iteration classification to compute loss on network scale_output
    transformation = model.get_initial_transform(1)
    for i in range(calibration_iters):
        image = model.extract_images_with_scales(x.unsqueeze(0), transformation, model.input_dims)
        arr = image.squeeze(0).detach().cpu().numpy()

        classification, transformation = model(image, history)

        plt.imsave(f"{image_name}_{i}.png", np.moveaxis(arr, 0, -1), format="png", vmin=0, vmax=1)
        print(f"[{image_name}] Classification {i}:", torch.argmax(classification).item())
    plt.imsave(f"{image_name}_original.png", inp.moveaxis(0, -1).numpy(), format="png", vmin=0, vmax=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--interpolate_size", type=int, default=8)
    parser.add_argument("--calibration_iters", type=int, default=4)
    parser.add_argument("--inference_filename", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # dataset = torchvision.datasets.mnist.MNIST(
    #     root="datasets/",
    #     download=True,
    #     transform=torchvision.transforms.ToTensor(),
    # )

    dataset = torchvision.datasets.PCAM(
        root="datasets/",
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    model = ScalelessViT(
        n_classes=2,
        input_dims=(args.interpolate_size, args.interpolate_size),
        transformer_history_size=args.calibration_iters,
        device=args.device,
        n_channels=dataset[0][0].shape[0],
        n_heads=args.calibration_iters
    )

    # Load data and model to device
    model = model.to(args.device)

    if args.inference_filename is not None:
        print("Writing inference images to file")
        model.load_state_dict(torch.load(args.inference_filename))
        for i in range(10):
            show_image_segments(model, dataset[i][0], args.calibration_iters,
                                image_name=f"images/im_{i}_{dataset[i][1]}")
        exit(0)

    # Split dataset
    tr = int(len(dataset) * 0.7)
    te = len(dataset) - tr
    train_dataset, test_dataset = data.random_split(dataset, lengths=(tr, te))

    # Create dataloaders
    test_dataloader = data.dataloader.DataLoader(test_dataset, batch_size=1024)
    train_dataloader = data.dataloader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")

        train(model, train_dataloader, args.calibration_iters)
        test(model, test_dataloader, args.calibration_iters)

        # Store model
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/model.chkpt")

    show_image_segments(model, dataset[0][0], args.calibration_iters)


if __name__ == "__main__":
    main()
