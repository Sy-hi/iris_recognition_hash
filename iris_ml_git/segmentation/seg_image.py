# Define a lightweight U-Net architecture
# Function:
# 1. Load pretrained U-Net for iris or pupil segmentation
# 2. Predict segmentation mask for a single image
# 3. Save result as RGBA PNG with transparent background

import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn

# U-Net architecture for segmentation
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = double_conv(n_channels, 64)
        self.encoder2 = double_conv(64, 128)
        self.encoder3 = double_conv(128, 256)
        self.encoder4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = double_conv(128, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.decoder4(torch.cat((enc4, self.upconv4(bottleneck)), dim=1))
        dec3 = self.decoder3(torch.cat((enc3, self.upconv3(dec4)), dim=1))
        dec2 = self.decoder2(torch.cat((enc2, self.upconv2(dec3)), dim=1))
        dec1 = self.decoder1(torch.cat((enc1, self.upconv1(dec2)), dim=1))

        return torch.sigmoid(self.final_conv(dec1))

# Predict and save a transparent PNG mask for a single image
def predict_single_image(input_image_path, output_image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image = Image.open(input_image_path).convert("RGB")
    orig_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    mask = output.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)

    orig_img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    if orig_img is None:
        raise RuntimeError(f"Failed to load image: {input_image_path}")
    if len(orig_img.shape) == 2:
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
    elif orig_img.shape[-1] == 4:
        orig_img = orig_img[:, :, :3]

    alpha_channel = mask.copy()
    b, g, r = cv2.split(orig_img)
    rgba = cv2.merge([b, g, r, alpha_channel])

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, rgba)
    print(f"Mask saved to: {output_image_path}")

# Run for single image
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_model.pth", map_location=device))
    print("U-Net model loaded.")

    input_image = r"C:\Users\User\Desktop\unsw\y2\9773\ELEC9773 Databases\Database 1 - CASIA v3.0-Iris-Interval\Images\001\L\S1001L03.jpg"
    output_image = r"C:\Users\User\Desktop\unsw\y2\9773\code\iris_ml\outputs\seg_overlay.png"

    predict_single_image(input_image, output_image, model, device)
