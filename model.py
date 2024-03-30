import torch

class ResidualBlock(torch.nn.Module):

    def __init__(self, base_channels):
        super(ResidualBlock, self).__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.GroupNorm(base_channels // 4, base_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_channels, base_channels, 3, padding=1, padding_mode='reflect'),
            
            torch.nn.GroupNorm(base_channels // 4, base_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_channels, base_channels, 3, padding=1, padding_mode='reflect'),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(torch.nn.Module):

    def __init__(self, base_channels: int = 32):
        super(Generator, self).__init__()

        self.conv1_1 = torch.nn.Conv2d(3, base_channels, 3, padding=1, padding_mode='reflect')
        self.activation_1 = torch.nn.ReLU(inplace=True)
        self.grp_norm_1 = torch.nn.GroupNorm(base_channels // 4, base_channels)

        self.downsample_2 = torch.nn.Conv2d(base_channels, base_channels *2, 3, stride=2, padding=1)
        self.activation_2 = torch.nn.ReLU(inplace=True)
        self.grp_norm_2 = torch.nn.GroupNorm(base_channels * 2 // 4, base_channels * 2)

        self.residual_blocks = torch.nn.Sequential(
            *[ResidualBlock(base_channels * 2) for _ in range(6)]
        )

        self.upsample_3 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_3 = torch.nn.Conv2d(base_channels * 2, 3, 3, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = self.grp_norm_1(self.activation_1(self.conv1_1(x)))
        x = self.grp_norm_2(self.activation_2(self.downsample_2(x)))
        x = self.residual_blocks(x)
        x = self.upsample_3(x)
        x = self.conv_3(x)
        x = torch.sigmoid(x)
        return x
    
class Discriminator(torch.nn.Module):

    def __init__(self, base_channels: int = 32):
        super(Generator, self).__init__()

        self.conv1_1 = torch.nn.Conv2d(3, base_channels, 3, padding=1, padding_mode='reflect')
        self.activation_1 = torch.nn.ReLU(inplace=True)
        self.grp_norm_1 = torch.nn.GroupNorm(base_channels // 4, base_channels)

        self.downsample_2 = torch.nn.Conv2d(base_channels, base_channels *2, 3, stride=2, padding=1)
        self.activation_2 = torch.nn.ReLU(inplace=True)
        self.grp_norm_2 = torch.nn.GroupNorm(base_channels * 2 // 4, base_channels * 2)

        self.residual_blocks = torch.nn.Sequential(
            *[ResidualBlock(base_channels * 2) for _ in range(2)]
        )
        # discrimnate whether the image is real or fake
        self.conv_3 = torch.nn.Conv2d(base_channels * 2, 1, 3, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = self.grp_norm_1(self.activation_1(self.conv1_1(x)))
        x = self.grp_norm_2(self.activation_2(self.downsample_2(x)))
        x = self.residual_blocks(x)
        x = self.conv_3(x)
        x = torch.sigmoid(x)
        return x


x = torch.randn(1, 3, 256, 256).cuda()
x = Generator(32).cuda()(x)