import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import torch.nn.functional as F

# ==================== 1. Dice Loss ==================== #
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # 确保输出在 0-1 之间
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice  # 让 Dice Loss 越小越好


def enhance_iris_image(image_rgb):
    """增强虹膜图像（CLAHE + 中值滤波）"""
    # 转 LAB 空间
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE 增强亮度通道
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    image_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # 中值滤波去除高频噪点/反光
    #image_filtered = cv2.medianBlur(image_eq, 3)

    return image_eq



# ==================== 2. 自定义数据集 ==================== #
class CustomDataset(Dataset):
    def __init__(self, img_folder, mask_folder, transform=None):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transform = transform

        self.img_list = sorted(self.get_all_images(img_folder, ['.jpg', '.jpeg', '.png', '.bmp']))
        print(f"✅ 发现 {len(self.img_list)} 张图片")

    def get_all_images(self, root_folder, valid_exts):
        return sorted([
            os.path.join(root, file)
            for root, _, files in os.walk(root_folder)
            for file in files if file.endswith(tuple(valid_exts))
        ])

    def preprocess_mask(self, mask):
        """确保掩码是平滑的弧形"""
        if mask is None:
            return None
        mask = (mask > 128).astype(np.uint8) * 255  # 二值化

        # **形态学闭运算（填补边界不平滑的问题）**
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # **高斯模糊（平滑边缘）**
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        return mask

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]

        # **支持 PNG 和 BMP 作为 mask**
        mask_filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path_png = os.path.join(self.mask_folder, mask_filename + ".png")
        mask_path_bmp = os.path.join(self.mask_folder, mask_filename + ".bmp")

        # **选择合适的 mask**
        mask_path = mask_path_png if os.path.exists(mask_path_png) else mask_path_bmp if os.path.exists(mask_path_bmp) else None
        if not mask_path:
            print(f"⚠️ 掩码文件缺失: {mask_filename}.png 或 {mask_filename}.bmp, 跳过")
            return None

        # 读取图片 & 掩码
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"⚠️ 读取失败: {img_path}, 跳过")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #新增对图像也进行高斯滤波处理
        #image = enhance_iris_image(image)


        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = self.preprocess_mask(mask)  # **平滑 mask**
        if mask is None:
            print(f"⚠️ 读取失败: {mask_path}, 跳过")
            return None

        # **检查 image 和 mask 形状匹配**
        if mask.shape[:2] != image.shape[:2]:
            print(f"❌ {img_path} 与 {mask_path} 形状不匹配, 跳过")
            return None

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        # **转换为 PyTorch Tensor**
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask).unsqueeze(0)  # **确保 `[1, H, W]` 格式**

        return image, mask


# ==================== 3. DataLoader 过滤 `None` ==================== #
def collate_fn(batch):
    """ 过滤掉 None 数据 """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# ==================== 3. U-Net 模型（动态调整大小） ==================== #
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

        dec4 = self.upconv4(bottleneck)
        dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=True)  # ✅ 修正
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=True)  # ✅ 修正
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=True)  # ✅ 修正
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=True)  # ✅ 修正
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)  # **输出 `n_classes` 个通道**

#计算f1 precision和recall
def compute_segmentation_metrics(preds, masks, threshold=0.5):
    preds_bin = (preds > threshold).float()
    TP = (preds_bin * masks).sum().item()
    FP = (preds_bin * (1 - masks)).sum().item()
    FN = ((1 - preds_bin) * masks).sum().item()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


# ==================== 4. 训练代码（不 `reshape`） ==================== #
import datetime

def train_model():
    epochs = 30
    batch_size = 12
    lr = 0.001
    final_lr = lr

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(
        r"C:\Users\User\Desktop\unsw\y2\9773\ELEC9773 Databases\Database 1 - CASIA v3.0-Iris-Interval\Images_1",
        r"C:\Users\User\Desktop\unsw\y2\9773\ELEC9773 Databases\Database 1 - CASIA v3.0-Iris-Interval\Segmentation groundtruth",
        transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ✅ 创建时间戳输出文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("system_predicted_outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            if batch is None:
                continue

            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(2)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"✅ Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    # ✅ 保存模型
    model_path = os.path.join(output_dir, "unet_model.pth")
    torch.save(model.state_dict(), model_path)

    # ✅ 评估精度指标（简单地用训练集评估，也可以替换为 val_loader）
    with torch.no_grad():
        model.eval()
        total_p, total_r, total_f1, count = 0, 0, 0, 0

        for batch in train_loader:
            if batch is None:
                continue
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(2)

            outputs = model(images)
            outputs_sigmoid = torch.sigmoid(outputs)

            p, r, f1 = compute_segmentation_metrics(outputs_sigmoid, masks)
            total_p += p
            total_r += r
            total_f1 += f1
            count += 1

        avg_p = total_p / count
        avg_r = total_r / count
        avg_f1 = total_f1 / count

    # ✅ 保存 metric 文档
    metric_path = os.path.join(output_dir, "metrics.txt")
    with open(metric_path, "w") as f:
        f.write(f"Segmentation Evaluation Metrics (Threshold=0.5)\n")
        f.write(f"Precision: {avg_p:.4f}\n")
        f.write(f"Recall:    {avg_r:.4f}\n")
        f.write(f"F1 Score:  {avg_f1:.4f}\n")
    print(f"📁 模型与评估结果已保存至: {output_dir}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    train_model()


