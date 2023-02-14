from functools import cache
import crop_interpolate
import torch
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
        encoder_layer = nn.TransformerEncoderLayer(latent_size, nhead=n_heads, dim_feedforward=128)
        self.transformer_model = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # self.transformer_model = Transformer(
        #     dim=latent_size, depth=4, heads=transformer_history_size,
        #     dim_head=n_heads, mlp_dim=latent_size)

        transformer_latent_size = latent_size * n_heads
        self.classifier_head = nn.Sequential(
            nn.Linear(transformer_latent_size, n_classes),
            nn.LogSoftmax(dim=-1),
        )
        self.transformation_head = nn.Sequential(
            nn.Linear(transformer_latent_size, 4),
            nn.Sigmoid()
        )

        self.profiler = cProfile.Profile()

        self.loss = nn.NLLLoss()
        self.transform_loss = nn.MSELoss()

        # TODO:
        # Try: Scale loss up to transformer instead of the entire image latent generation part as well
        # Try: KL divergence loss for the scale loss
        # Try: Binary classification for scale loss

    @cache
    def get_initial_transforms(self, batch_size):
        """
        Initial transformation to be used for patch extraction.
        The initial patches should encapsulate the entire image, and take up the entire width and height.
        The transformation will return a tensor of shape [B, 4] with values [0, 0, 1, 1]
        These values indicate an X- and Y-offset of 0%, and an X- and Y-width of 100%.

        :param batch_size:
        :return:
        """
        # Initial transformation is 0, 0 (start in left upper corner) and 1, 1 (use the entire image size)
        initial = torch.stack([torch.FloatTensor([0, 0, 1, 1]) for i in range(batch_size)])
        zeros = torch.zeros(initial.shape, device=self.device, requires_grad=True)
        return initial.to(self.device), zeros

    def get_initial_history(self, batch_size):
        """
        Generate a history list with correctly shaped zero-tensors to be used as padding in the first iterations of the
         model.

        :param batch_size:
        :return:
        """
        return [torch.zeros((batch_size, self.latent_size), device=self.device, requires_grad=False) for _ in
                range(self.transformer_history_size)]

    def compute_loss(self, classes, p_classes, transformation, p_transform, targets):
        """
        Compute two losses for the classification and for the transformation values.
        The classification loss is simply based on the accuracy on the targets.
        The transformation loss is computed with a combination of the classification and transformation.
        We compute the transformation loss relative to p_transform, whereas we compute the classification loss relative
         to the current classification.

        Transformation loss example in a binary classification task:
        ```
        targets     (y) = [1]
        classes     (c) = [0.6, 0.4]
        p_classes   (C) = [0.7, 0.3]
        transform   (t) = [0.1, 0.1, 0.5, 0.5]
        p_transform (T) = [0.2, 0.2, 0.3, 0.3]

        c - C             = [-0.1, 0.1]
        s = (c - C)[y]    = [0.1]

        t - T             = [-0.1, -0.1, 0.2, 0.2]
        d = (t - T) * s   = [-0.01, -0.01, 0.02, 0.02]
        target = T + d    = [0.19, 0.19, 0.32, 0.32]

        transformation_loss = MSELoss(p_transform, target)
        ```

        :param classes: current classification
        :param p_classes: previous classification
        :param transformation: current transformation
        :param p_transform: previous transformation
        :param targets: classification targets
        :return: classification_loss, transformation_loss
        """
        # Calculate loss on the classification
        class_loss = self.loss(classes, targets)

        # Compute how much better the classification was as a result of the new transformation:
        # if   p_classes -> classes   went up, the change from   p_transform -> transform   was positive
        improvement = torch.gather(classes - p_classes, 1, targets.unsqueeze(-1))
        transform_target = p_transform + ((transformation - p_transform) * improvement)

        transform_loss = self.transform_loss(p_transform, transform_target)

        return class_loss, transform_loss

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
        transformer_input = torch.stack(history[len(history) - (self.transformer_history_size - 1):] + [image_latent],
                                        dim=1)
        history.append(image_latent.detach())

        transformer_latent = (
            self.transformer_model(transformer_input).reshape(b, self.transformer_history_size * self.latent_size)
        )

        class_pred = self.classifier_head(transformer_latent)
        transform_pred = self.transformation_head(transformer_latent)

        return class_pred, transform_pred

    def extract_images_with_scales(self, x: torch.IntTensor, scales: torch.FloatTensor,
                                   dims: torch.IntTensor) -> torch.IntTensor:
        return crop_interpolate.crop_interpolate(
            x.to(self.device),
            scales.to(self.device),
            dims
        )