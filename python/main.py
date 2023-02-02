import os
from typing import List
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from model import ScalelessViT
import torchvision.datasets.mnist
from torch.utils import data


def train(model, train_dataloader, calibration_iters):
    optimizer = torch.optim.Adam(lr=2e-4, params=model.parameters())

    bar = tqdm(train_dataloader)
    for x, targets in bar:
        targets = targets.to(model.device)
        x = x.to(model.device)
        bs = targets.shape[0]

        optimizer.zero_grad()

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
            classification, transformation = model(images, history)

            loss = model.compute_loss(classification, previous_classes, targets=targets)
            loss.backward(retain_graph=True)

            # Store previous classes to get loss next iteration
            previous_classes = classification

            # Report loss improvement over refinement iterations
            if i == 0:
                start_loss = loss.item()
            end_loss = loss.item()
            bar.set_postfix_str(f"Loss: {round(start_loss, 2)} -> {round(end_loss, 2)}")

        # model.profiler.print_stats()
        # exit(0)
        optimizer.step()


def test(model, test_dataloader, calibration_iters):
    losses = []
    accs = []
    losses_1s = []
    accs_1s = []

    bar = tqdm(test_dataloader)
    model.eval()
    for x, targets in bar:
        x = x.to(model.device)
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

            loss = model.compute_loss(classification, previous_classes, targets=targets)

            # Store previous classes to get loss next iteration
            previous_classes = classification

            if i == 0:
                accs_1s.append(torch.mean((torch.argmax(previous_classes, dim=1) == targets).float()))
                losses_1s.append(loss.item())

            losses.append(loss.item())
        accs.append(torch.mean((torch.argmax(previous_classes, dim=1) == targets).float()))
    model.train()
    print(f"Val loss: {sum(losses) / len(losses)} (1-shot: {sum(losses_1s) / len(losses_1s)})")
    print(f"Val acc: {sum(accs) / len(accs)} (1-shot: {sum(accs_1s) / len(accs_1s)}")


def show_image_segments(model: ScalelessViT, x, calibration_iters):
    # History for this batch
    history = model.get_initial_history(1)

    x = x.to(model.device)

    # Store previous iteration classification to compute loss on network scale_output
    transformation = model.get_initial_transform(1)
    for i in range(calibration_iters):
        image = model.extract_images_with_scales(x.unsqueeze(0), transformation, model.input_dims)
        classification, transformation = model(image, history, return_image=True)
        arr = image.to("cpu").detach().numpy()
        plt.imsave(f"image_{i}.png", arr.reshape((8, 8)), format="png")

    print("Classification:", torch.argmax(classification))


def main():
    device = "cuda:0"
    batch_size = 64
    n_epochs = 10
    calibration_iters = 8  # How many subsections of the image will we look at

    model = ScalelessViT(n_classes=10, input_dims=(8, 8), transformer_history_size=calibration_iters, device=device)

    dataset = torchvision.datasets.mnist.MNIST(
        root="datasets/",
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    # Load data and model to device
    model = model.to(device)
    dataset.data.to(device)
    dataset.targets.to(device)

    inference = False
    if inference:
        model.load_state_dict(torch.load("checkpoints/model.chkpt"))
        print(dataset[3][1])
        show_image_segments(model, dataset[3][0], 10)
        exit(0)

    # Split dataset
    tr = int(len(dataset) * 0.7)
    te = len(dataset) - tr
    train_dataset, test_dataset = data.random_split(dataset, lengths=(tr, te))

    # Create dataloaders
    test_dataloader = data.dataloader.DataLoader(test_dataset, batch_size=1024)
    train_dataloader = data.dataloader.DataLoader(train_dataset, batch_size=batch_size)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")

        train(model, train_dataloader, calibration_iters)
        test(model, test_dataloader, calibration_iters)

    # Store model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model.chkpt")

    show_image_segments(model, dataset[0][0], 10)


if __name__ == "__main__":
    main()
