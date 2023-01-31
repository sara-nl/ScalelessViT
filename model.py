import torch
from torch import nn
from layers.transformer import Transformer


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
            ResidualBlock(3, 1),
            ResidualBlock(1, 1),
            ResidualBlock(1, 1),
            ResidualBlock(1, 1),
            ResidualBlock(1, 1),
            ResidualBlock(1, 1),
            ResidualBlock(1, 1),
        )

    def forward(self, x):
        return self.model(x)


class ScalelessViT(nn.Module):
    def __init__(self, n_classes=3, input_dims=(64, 64), latent_size=32, transformer_history_size=8):
        super().__init__()

        self.input_dims = input_dims
        self.image_model = nn.Sequential(
            ResNet18(),
            nn.Flatten(),
            nn.Linear(64 * 64, latent_size)
        )

        self.transformer_model = Transformer(
            dim=latent_size, depth=4, heads=transformer_history_size,
            dim_head=latent_size, mlp_dim=latent_size)

        transformer_latent_size = latent_size
        self.classifier_head = nn.Sequential(
            nn.Linear(transformer_latent_size, n_classes),
        )
        self.transformation_head = nn.Sequential(
            nn.Linear(transformer_latent_size, 4),
            nn.Sigmoid()
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, transformation, history):
        """
        x should be a list of image tensors, the images need not be the same shapes.
        Pass input, scales, and previous image_model latents.
        Scale is defined as [offset_x, offset_y, scale_x, scale_y]

        :param x: Batch of images of any size [b, c, x, y]
        :param transformation: Batch of scales for each image [b, 4]
        :param history: List of previous transformer latents, the list will get 1 new entry
        :return:
        """
        # Create
        images = self._extract_images_with_scales(x, transformation)

        image_latent = self.image_model(images)

        # Create a stacked latent history tensor
        history.append(image_latent.detach())
        transformer_input = torch.stack(history, dim=1)

        transformer_latent = self.transformer_model(transformer_input)

        class_pred = self.classifier_head(transformer_latent)
        transform_pred = self.transformation_head(transformer_latent)

        return class_pred, transform_pred

    def _extract_images_with_scales(self, x, scales):
        sampled_images = []
        for image, scale in zip(x, scales):
            # Extract percentage x,y and zoom x,z
            px, py = scale[:2]
            zx, zy = scale[2:]

            w, h = image.shape[1:]  # Ignore channel

            # Get the selected box starting point as subset of the image.
            # We now scale the   0,1   range to    0,(size-64)  so we cannot have invalid percentages.
            x1 = px * (w - self.input_dims[0])
            y1 = py * (h - self.input_dims[1])

            # We scale the image to be 64 for zoom=0, and image_size for zoom=1
            x_size = (zx * (w - self.input_dims[0])) + self.input_dims[0]
            y_size = (zy * (h - self.input_dims[1])) + self.input_dims[1]

            # Clip max size based on available remaining pixels
            x_size = min(w - x1, x_size)
            y_size = min(h - y1, y_size)

            sampled_images.append(torch.nn.functional.interpolate(
                image.unsqueeze(0)[:, :, x1:x1 + x_size, y1:y1 + y_size],
                self.input_dims, mode="nearest"
            ))

        return torch.stack(sampled_images)
