import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from dataset import VOCDataset
from transfer import TransferNet
from optimizer import VGGNet
from compute_loss import content_loss, style_loss
from utils import load_image


# path & Hyper parameters
batch_size = 8
learning_rate = 1e-4
style_weight = 1e-7
content_weight = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
content_image_dir = "F:\GithubRepository\图像分割数据集\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages"
style_image_path = "styles/rain_princess.jpg"
save_path = "outputs\\"

# 数据集
# content
dataset = VOCDataset(content_image_dir)
dataloader = DataLoader(dataset, batch_size, shuffle=True)
# style
style_image = load_image(style_image_path, (256, 256))
transform = transforms.Compose([
    transforms.ToTensor()
])
style_image = transform(style_image).unsqueeze(0).to(device)

# model
transfer_net = TransferNet().to(device).train()
feature_extractor = VGGNet().to(device).eval()

# 优化器
opt = optim.Adam(transfer_net.parameters(), lr=learning_rate)

# train
style_features = feature_extractor(style_image)  # 提取风格图特征

while True:
    for idx, content_image in enumerate(dataloader):
        content_image = content_image.to(device)
        combination = transfer_net(content_image)
        # 计算loss
        content_features = feature_extractor(content_image)  # 提取内容图特征
        combination_features = feature_extractor(combination)  # 提取生成图特征
        # 风格loss
        s_loss = 0
        for i in range(len(style_features)):
            s_loss += style_loss(style_features[i].detach(), combination_features[i], batch_size)
        # 内容loss
        c_loss = content_loss(content_features[4].detach(), combination_features[4])
        # 总loss
        loss = style_weight * s_loss + content_weight * c_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"iter:{idx}, loss:{loss.item()}, s_loss:{s_loss.item()}, c_loss:{c_loss.item()}")
        if idx % 100 == 0:
            torch.save(transfer_net.state_dict(), 'fst.pth')
            save_image([content_image[0], combination[0]], save_path+f"iter_{idx}.png")







