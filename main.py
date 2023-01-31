from typing import List
import torch

from model import ScalelessViT


def main():
    batch_size = 1
    model = ScalelessViT(n_classes=3)

    calibration_iters = 10  # How many subsections of the image will we look at

    dataset: List[torch.Tensor] = [
        (torch.rand((1, 1, 412, 122)), torch.ones((1, 1), dtype=int)),
        (torch.rand((1, 1, 1514, 321)), torch.ones((1, 1), dtype=int)),
    ]

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

            print(classification.shape, transformation.shape)
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
    main()
