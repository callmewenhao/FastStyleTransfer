# TODO loss计算器
import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg19


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = vgg19(pretrained=True).features

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


def main():
    feature_extractor = VGGNet()


if __name__ == "__main__":
    main()





