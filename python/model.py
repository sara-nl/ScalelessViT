from functools import cache
from typing import Tuple

import torch
import torchvision.transforms.functional_tensor
from torch import nn
import cProfile


class ResidualBlock(nn.Module):
    """
    Taken from
    https://github.com/FrancescoSaverioZuppichini/ResNet/blob/master/ResNet.ipynb
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x if not self.should_apply_shortcut else self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNet18(nn.Module):
    """
    Doe maar hierin thomas
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            ResidualBlock(3, 32),
            ResidualBlock(32, 32),
        )

    def forward(self, x):
        return self.model(x)


class ScalelessViT(nn.Module):
    def __init__(self, n_classes=3, input_dims=(64, 64), latent_size=32, transformer_history_size=8,
                 n_heads=8, device="cpu", n_channels=1):
        super().__init__()

        self.device = device
        self.n_classes = n_classes
        self.input_dims: torch.IntTensor = torch.IntTensor(input_dims)
        self.transformer_history_size = transformer_history_size
        self.latent_size = latent_size

        image_size = n_channels
        for dim in input_dims:
            image_size *= dim

        # Assume pretrained
        self.image_model = nn.Sequential(
            ResNet18(),
            nn.Flatten(),
            nn.Linear(image_size, latent_size)
        )

        encoder_layer = nn.TransformerEncoderLayer(latent_size, nhead=n_heads, dim_feedforward=32)
        self.transformer_model = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # self.transformer_model = Transformer(
        #     dim=latent_size, depth=4, heads=transformer_history_size,
        #     dim_head=n_heads, mlp_dim=latent_size)

        transformer_latent_size = latent_size * n_heads
        self.classifier_head = nn.Sequential(
            nn.Linear(transformer_latent_size, n_classes),
            nn.LogSoftmax(),
        )
        self.transformation_head = nn.Sequential(
            nn.Linear(transformer_latent_size, 4),
            nn.Sigmoid()
        )

        self.profiler = cProfile.Profile()

        self.loss = nn.NLLLoss()

        # TODO:
        # Try: Scale loss up to transformer instead of the entire image latent generation part as well
        # Try: KL divergence loss for the scale loss
        # Try: Binary classification for scale loss

    @staticmethod
    @cache
    def get_initial_transform(batch_size):
        # Initial transformation is 0, 0 (start in left upper corner) and 1, 1 (use the entire image size)
        return torch.stack([torch.FloatTensor([0, 0, 1, 1]) for i in range(batch_size)])

    def get_initial_history(self, batch_size):
        return [torch.zeros((batch_size, self.latent_size), device=self.device, requires_grad=False) for _ in
                range(self.transformer_history_size)]

    def compute_loss(self, classification, previous_classes, targets):
        # Calculate loss on the classification
        cl_weight = 0.5  # Class loss weight
        sl_weight = 0.5  # Scale loss weight, use to balance loss combination
        class_loss = self.loss(classification, targets) * cl_weight

        # Scale loss subtracts previous prediction from current.
        # If previous prediction on target class was better, it should have higher loss
        # If current prediction is better, loss should be lower
        # norms = norm(classification - previous_classes)
        scale_loss = self.loss(classification - previous_classes, targets) * sl_weight

        return class_loss, scale_loss

    def forward(self, x, history):
        """
        x should be a list of image tensors, the images need not be the same shapes.
        Pass input, scales, and previous image_model latents.
        Scale is defined as [offset_x, offset_y, scale_x, scale_y]

        :param x: Batch of images of any size [b, c, w, h]
        :param transformation: Batch of scales for each image [b, 4]
        :param history: List of previous transformer latents, the list will get 1 new entry
        :param return_image:
        :return:
        """
        b, c, w, h = x.shape

        # Assume a pretrained image model exists.
        image_latent = self.image_model(x)

        # Store latents to use for future iterations
        # History is in format: N, B, D
        # Reformat to B, N, D before using

        # We will only train the transformer model, so creating the input starts here
        # The history array is of shape [n, b, l]
        # N is the nth iteration, b is the batch size, and l is the latent size.

        # Take the last N samples from history to build transformer input
        transformer_input = torch.stack(history[-(self.transformer_history_size - 1):] + [image_latent], dim=1)
        history.append(image_latent.detach())

        transformer_latent = (
            self.transformer_model(transformer_input).reshape(b, self.transformer_history_size * self.latent_size)
        )

        class_pred = self.classifier_head(transformer_latent)
        transform_pred = self.transformation_head(transformer_latent)

        return class_pred, transform_pred

    def extract_images_with_scales(self, x: torch.IntTensor, scales: torch.FloatTensor,
                                   dims: torch.IntTensor) -> torch.IntTensor:
        import crop_interpolate
        return crop_interpolate.crop_interpolate(
            x.to(self.device),
            scales.to(self.device),
            dims
        )

        sampled_images = []
        for i in range(len(x)):
            scale = scales[i]
            image = x[i]

            # Extract percentage x,y and zoom x,y
            px, py, zx, zy = [scale[n].item() for n in range(4)]
            c, w, h = image.shape

            avx = (w - dims[0])  # Available space X
            avy = (h - dims[1])  # Available space Y

            # Get the selected box starting point as subset of the image.
            # We now scale the   0,1   range to    0,(size-64)  so we cannot have invalid percentages.
            x1 = px * avx
            y1 = py * avy

            # We scale the image to be 64 for zoom=0, and image_size for zoom=1
            # Clip max size based on available remaining pixels
            x_size = min(w - x1, (zx * avx) + dims[0])
            y_size = min(h - y1, (zy * avy) + dims[1])

            cropped = torchvision.transforms.functional_tensor.crop(
                image, int(x1), int(y1), int(x_size), int(y_size)
            ).unsqueeze(0)
            interpolated = torch.nn.functional.interpolate(
                cropped,
                dims[0].item(), mode="nearest"
            ).squeeze(0)
            sampled_images.append(interpolated)

        return torch.stack(sampled_images).to(self.device)
