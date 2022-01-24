# TODO 风格转换器
import torch
import torch.nn as nn

# input 3x256x256 output 3x256x256


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, r=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels)
        )
        self.r = r

    def forward(self, x):
        x = self.conv(x)
        if self.r:
            return torch.relu(x)
        else:
            return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, 1, 1)
        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size, 1, 1, r=False)

    def forward(self, x):
        x = self.conv1(x)
        return x + self.conv2(x)


class ConvTransposeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_transpose(x)


class TransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 32, 9, 1, 4)
        self.conv2 = ConvLayer(32, 64, 3, 2, 1)
        self.conv3 = ConvLayer(64, 128, 3, 2, 1)
        self.residual = ResidualBlock(128, 128, 3)
        self.conv_transpose1 = ConvTransposeLayer(128, 64, 2, 2)
        self.conv_transpose2 = ConvTransposeLayer(64, 32, 2, 2)
        self.conv4 = ConvLayer(32, 3, 9, 1, 4, r=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        for i in range(5):
            x = self.residual(x)
        x = self.conv_transpose1(x)
        x = self.conv_transpose2(x)
        x = self.conv4(x)
        return self.sigmoid(x)


def main():
    x = torch.randn(1, 3, 256, 256)
    print(f"input shape is {x.shape}")
    model = TransferNet()
    out = model(x)
    print(f"output shape is {out.shape}")
    assert x.shape == out.shape


if __name__ == "__main__":
    main()



















