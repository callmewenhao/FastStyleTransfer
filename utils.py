import os

import torch
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path, new_size=(256, 256)):  # 宽 高
    img = Image.open(image_path)
    img = img.resize(new_size)
    return img


def main():
    image_path = "F:\GithubRepository\图像分割数据集\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages\\2007_000027.jpg"
    img = load_image(image_path)
    plt.figure()
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # main()
    loss = torch.zero



