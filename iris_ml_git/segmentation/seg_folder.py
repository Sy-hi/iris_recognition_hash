# Define a lightweight U-Net architecture
# Function:
# 1. Predict segmentation masks for input images
# 2. Save results as RGBA PNGs with transparent backgrounds
# 3. Recursively process entire folders of images
# Used in iris or pupil segmentation tasks with pretrained models.

import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
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
        # Encoding path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoding path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))


# Predict and save a transparent PNG mask for a single image
def predict_and_save(input_image_path, output_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    try:
        image = Image.open(input_image_path).convert("RGB")
    except Exception as e:
        print(f"Cannot open image: {input_image_path}, Error: {e}")
        return

    orig_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    # Postprocess output mask
    mask = output.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)

    orig_img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    if orig_img is None:
        print(f"Failed to load image with OpenCV: {input_image_path}, skipping.")
        return

    if len(orig_img.shape) == 2:
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
    elif orig_img.shape[-1] == 4:
        orig_img = orig_img[:, :, :3]

    # Add alpha channel
    alpha_channel = mask.copy()
    b, g, r = cv2.split(orig_img)
    rgba = cv2.merge([b, g, r, alpha_channel])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, rgba)


# Process all images in a folder and save predictions
def process_folder(input_folder, output_folder, model, device):
    os.makedirs(output_folder, exist_ok=True)

    image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))

    if len(image_paths) == 0:
        print(f"Skipped empty folder: {input_folder}")
        return

    print(f"Found {len(image_paths)} images. Processing...")

    for input_path in tqdm(image_paths, desc="Processing", unit="img"):
        relative_path = os.path.relpath(input_path, input_folder)
        output_path = os.path.join(output_folder, relative_path).replace(".jpg", ".png").replace(".jpeg", ".png")
        predict_and_save(input_path, output_path, model, device)

    print(f"All done. Results saved to: {output_folder}")


# Run the segmentation pipeline
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_model.pth", map_location=device))
    print("U-Net model loaded.")

    input_folder = r"C:\Users\User\Desktop\unsw\y2\9773\ELEC9773 Databases\Database 1 - CASIA v3.0-Iris-Interval\Images"
    output_folder = r"C:\Users\User\Desktop\unsw\y2\9773\ELEC9773 Databases\Database 1 - CASIA v3.0-Iris-Interval\Predicted_1"

    process_folder(input_folder, output_folder, model, device)
