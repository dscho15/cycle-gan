import torch


class ConvBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        act: bool = True,
        norm: bool = True,
    ):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            padding_mode="reflect",
        )
        self.norm = (
            torch.nn.GroupNorm(out_channels // 2, out_channels)
            if norm
            else torch.nn.Identity()
        )
        self.act = torch.nn.ReLU() if act else torch.nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(torch.nn.Module):

    def __init__(self, base_channels):
        super(ResidualBlock, self).__init__()

        self.conv_block = torch.nn.Sequential(
            ConvBlock(base_channels, base_channels, 3, 1, 1, act=True),
            ConvBlock(base_channels, base_channels, 3, 1, 1, act=False),
        )

    def forward(self, x: torch.Tensor):
        return x + self.conv_block(x)


class Generator(torch.nn.Module):

    def __init__(self, base_channels: int = 64, n_residual_blocks: int = 9):
        super(Generator, self).__init__()

        self.down_blocks = torch.nn.Sequential(
            ConvBlock(3, base_channels, 7, 1, 3),
            ConvBlock(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            ConvBlock(
                base_channels * 2,
                base_channels * 4,
                kernel_size=3,
                stride=2,
                padding=1
            ),
        )

        self.middle_blocks = torch.nn.Sequential(
            *[ResidualBlock(base_channels * 4) for _ in range(n_residual_blocks)]
        )

        self.up_blocks = torch.nn.Sequential(
            ConvBlock(
                base_channels * 4,
                base_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                act=True,
                norm=True,
            ),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(
                base_channels * 2,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                act=True,
                norm=True,
            ),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(
                base_channels,
                3,
                kernel_size=7,
                stride=1,
                padding=3,
                act=False,
                norm=False,
            ),
        )

    def forward(self, x: torch.Tensor):
        x = self.down_blocks(x)
        x = self.middle_blocks(x)
        x = self.up_blocks(x)
        return torch.tanh(x)


class Discriminator(torch.nn.Module):

    def __init__(self, base_channels: int = 64, wasserstein: bool = False):
        super(Discriminator, self).__init__()

        # convert into a sequential
        self.model = torch.nn.Sequential(

            torch.nn.Conv2d(
                3, base_channels, 4, stride=2, padding=1, padding_mode="reflect"
            ),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(
                base_channels, base_channels * 2, 4, stride=2, padding_mode="reflect"
            ),
            torch.nn.GroupNorm(base_channels // 2, base_channels * 2),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                4,
                stride=2,
                padding_mode="reflect",
            ),
            torch.nn.GroupNorm(base_channels, base_channels * 4),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(
                base_channels * 4,
                base_channels * 8,
                4,
                stride=2,
                padding_mode="reflect",
            ),
            torch.nn.GroupNorm(base_channels * 2, base_channels * 8),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(
                base_channels * 8, 1, kernel_size=4, stride=1, padding_mode="reflect"
            ),
        )

        self.wasserstein = wasserstein

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return torch.sigmoid(x) if not self.wasserstein else x
