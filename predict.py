import torch
from transfer import TransferNet
from utils import load_image
import torchvision.transforms as transforms
from torchvision.utils import save_image

# path
content_path = "content/chicago.jpg"
style_path = "styles/rain_princess.jpg"
save_dir = "outputs\\"
weight_path = "fst.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# image
content = load_image(content_path, new_size=(256, 256))
style = load_image(style_path, new_size=(256, 256))
s_data = transform(style).to(device)
c_data = transform(content).unsqueeze(0).to(device)

# model
model = TransferNet().to(device)
model.load_state_dict(torch.load(weight_path))

# pred
model.eval()
with torch.no_grad():
    out = model(c_data)
save_image([s_data, c_data[0], out[0]], save_dir+f"pred0.png")


