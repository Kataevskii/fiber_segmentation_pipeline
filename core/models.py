import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv3D -> GroupNorm/BatchNorm -> GELU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        # Use GroupNorm for small batch stability
        groups = min(num_groups, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with MaxPool3D then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    """Upscaling then DoubleConv"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2 size
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class IntensityUNet3D(nn.Module):
    """
    Dedicated 3D U-Net Specialist for Centerline Probability & Radial Gaussian Intensity Potential Field.
    Input: (B, 1, D, H, W) Raw grayscale microscopy volume.
    Output: (B, 1, D, H, W) Continuous signed potential field in [-1, 1].
    """
    def __init__(self, in_channels=1, base_channels=24, trilinear=True):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)

        self.up1 = Up(c * 8 + c * 4, c * 4, trilinear)
        self.up2 = Up(c * 4 + c * 2, c * 2, trilinear)
        self.up3 = Up(c * 2 + c, c, trilinear)

        # Output head: predicts continuous signed potential field in [-1, 1]
        self.out_conv = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        raw_intensity = self.out_conv(x)
        intensity_out = torch.tanh(raw_intensity)
        return intensity_out


class OrientationUNet3D(nn.Module):
    """
    Dedicated 3D U-Net Specialist for Continuous 3D Tangent Vector Orientation Field.
    Input: (B, 1, D, H, W) Raw grayscale microscopy volume.
    Output: (B, 3, D, H, W) Unit normalized 3D vectors (vz, vy, vx).
    """
    def __init__(self, in_channels=1, base_channels=24, trilinear=True):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)

        self.up1 = Up(c * 8 + c * 4, c * 4, trilinear)
        self.up2 = Up(c * 4 + c * 2, c * 2, trilinear)
        self.up3 = Up(c * 2 + c, c, trilinear)

        # Output head: predicts 3 vector components (vz, vy, vx)
        self.out_conv = nn.Conv3d(c, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        raw_dir = self.out_conv(x)
        # Strictly normalize vectors to unit length
        norm = torch.linalg.norm(raw_dir, dim=1, keepdim=True) + 1e-8
        dir_out = raw_dir / norm
        return dir_out
