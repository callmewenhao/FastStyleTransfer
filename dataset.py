import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils import load_image


class VOCDataset(Dataset):
    def __init__(self, image_dir, new_size=(256, 256)):
        super().__init__()
        self.image_dir = image_dir
        self.new_size = new_size
        self.image_names = os.listdir(image_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, self.image_names[item])
        img = load_image(image_path, self.new_size)  # 256 256
        return self.transform(img)


def main():
    image_dir = "F:\GithubRepository\图像分割数据集\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages"
    dataset = VOCDataset(image_dir, (256, 256))
    print(len(dataset))
    image = dataset[0]
    print(image.shape)


if __name__ == "__main__":
    main()




















