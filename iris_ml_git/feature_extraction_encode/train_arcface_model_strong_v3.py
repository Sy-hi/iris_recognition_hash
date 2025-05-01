# Main training pipeline for iris recognition using ArcFace + hash encoding + triplet loss
# Includes training, validation, threshold analysis, t-SNE visualization, and encoding export

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow multiple OpenMP instances on some platforms

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from arcface_iris_model import ArcFaceIrisNet
from losses_v2 import MultiTaskIrisLoss
from center_loss import CenterLoss
from save_encode_analyze.evaluate_accuracy_and_hamming_v4 import evaluate_accuracy_and_hamming
from save_encode_analyze.save_encodings_and_analyze import save_encodings, compute_hamming_statistics, recognition_accuracy
from save_encode_analyze.threshold_search import search_best_threshold
from save_encode_analyze.tsne_visualizer import visualize_tsne
from save_encode_analyze.save_image_encodings_both import save_image_encodings


# Dataset class for triplet sampling
class TripletLabelDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
        self.label_to_paths = {}
        for path, label in self.data:
            self.label_to_paths.setdefault(label, []).append(path)
        self.labels = list(self.label_to_paths.keys())

    def __getitem__(self, index):
        anchor_path, anchor_label = self.data[index]
        positive_path = random.choice(self.label_to_paths[anchor_label])
        while positive_path == anchor_path and len(self.label_to_paths[anchor_label]) > 1:
            positive_path = random.choice(self.label_to_paths[anchor_label])
        negative_label = random.choice([l for l in self.labels if l != anchor_label])
        negative_path = random.choice(self.label_to_paths[negative_label])

        def load(path): return self.transform(Image.open(path).convert("L"))

        return load(anchor_path), load(positive_path), load(negative_path), anchor_label

    def __len__(self):
        return len(self.data)


# Split dataset into train and validation sets by ID and eye side
def split_dataset_by_id(root_dir, train_ratio=0.8):
    train_paths, val_paths = [], []
    for person_id in os.listdir(root_dir):
        for eye in ["L", "R"]:
            eye_id = f"{person_id}_{eye}"
            subfolder = os.path.join(root_dir, person_id, eye)
            if not os.path.isdir(subfolder):
                continue
            image_files = [f for f in os.listdir(subfolder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                continue
            full_paths = [os.path.join(subfolder, f) for f in image_files]
            random.shuffle(full_paths)
            split = max(1, int(len(full_paths) * train_ratio))
            train_paths += [(f, eye_id) for f in full_paths[:split]]
            val_paths += [(f, eye_id) for f in full_paths[split:]]
    return train_paths, val_paths


# Create a timestamped output directory
def make_output_dir(base="feature_extraction_encoding_result"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# Training loop with evaluation and metric logging
def train(model, dataloader, val_paths, optimizer, criterion, device, transform,
          output_dir, class_to_idx, epochs=30, hamming_threshold=25):
    best_acc = 0
    log_csv = os.path.join(output_dir, "log.csv")
    loss_list, val_acc_list = [], []

    # Center loss with separate optimizer
    center_criterion = CenterLoss(num_classes=len(class_to_idx), feat_dim=model.hash_dim, device=device).to(device)
    center_optimizer = optim.SGD(center_criterion.parameters(), lr=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for anchor, positive, negative, labels in loop:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            labels = torch.tensor(labels).to(device)

            anchor_hash, anchor_logits = model(anchor, labels)
            pos_hash, _ = model(positive, None)
            neg_hash, _ = model(negative, None)

            total_loss, arc_loss, hash_loss, triplet_loss, center_val = criterion(
                anchor_logits, labels, anchor_hash, anchor_hash, pos_hash, neg_hash
            )

            center_loss_val = center_criterion(anchor_hash, labels)
            total_loss += 0.01 * center_loss_val  # Weighted center loss

            optimizer.zero_grad()
            center_optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            center_optimizer.step()

            loop.set_postfix(
                total=total_loss.item(),
                arc=arc_loss,
                hash=hash_loss,
                triplet=triplet_loss,
                center=center_loss_val.item()
            )

        avg_loss = total_loss / len(dataloader)

        # Validation phase
        val_acc = evaluate_accuracy_and_hamming(
            model, val_paths, transform, device,
            save_csv=os.path.join(output_dir, "match_log.csv"),
            save_fig=os.path.join(output_dir, "hamming_hist.png"),
            match_csv=os.path.join(output_dir, "match_log.csv"),
            hamming_threshold=hamming_threshold)

        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, val_acc={val_acc:.2f}%")
        loss_list.append(avg_loss)
        val_acc_list.append(val_acc)

        # Save log to CSV
        with open(log_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(["epoch", "loss", "val_acc", "lr"])
            writer.writerow([epoch + 1, avg_loss, val_acc, optimizer.param_groups[0]['lr']])

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

    # Post-training analysis
    visualize_tsne(model, val_paths, transform, device, save_path=os.path.join(output_dir, "tsne_hash.png"))
    search_best_threshold(model, val_paths, transform, device, output_dir)

    # Save encodings and evaluate distance distributions
    train_enc = save_encodings(model, train_set, transform, device,
                               os.path.join(output_dir, "train_encodings.csv"),
                               os.path.join(output_dir, "train_encodings.pkl"))
    val_enc = save_encodings(model, val_set, transform, device,
                             os.path.join(output_dir, "val_encodings.csv"),
                             os.path.join(output_dir, "val_encodings.pkl"))

    best_thresh, intra_range, inter_range = compute_hamming_statistics(
        train_enc,
        save_path=os.path.join(output_dir, "hamming_stats.csv"),
        plot_path=os.path.join(output_dir, "hamming_hist.png"),
        thresh_txt_path=os.path.join(output_dir, "threshold.txt")
    )

    print("Intra Range:", intra_range, "| Inter Range:", inter_range)
    print("Best Threshold:", best_thresh)

    final_acc = recognition_accuracy(val_enc, train_enc, threshold=best_thresh)
    print(f"Final Validation Accuracy (1:N): {final_acc:.2f}%")

    # Save image encodings to CSV/PKL

    save_image_encodings(model, image_paths=train_set, transform=transform, device=device,
                         csv_path=os.path.join(output_dir, "train_encodings.csv"),
                         pkl_path=os.path.join(output_dir, "train_encodings.pkl"))
    save_image_encodings(model, image_paths=val_set, transform=transform, device=device,
                         csv_path=os.path.join(output_dir, "val_encodings.csv"),
                         pkl_path=os.path.join(output_dir, "val_encodings.pkl"))

    # Save training curve
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "training_curve.png"))
    plt.close()


# Entry point for training
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = r"C:\Users\User\Desktop\unsw\y2\9773\ELEC9773 Databases\Database 1 - CASIA v3.0-Iris-Interval\after_process\Normalized_half"

    transform = transforms.Compose([
        transforms.Resize((32, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    output_dir = make_output_dir()
    train_set, val_set = split_dataset_by_id(dataset_path)
    all_ids = sorted(set([label for _, label in train_set + val_set]))
    class_to_idx = {cls: i for i, cls in enumerate(all_ids)}

    train_data = [(path, class_to_idx[label]) for path, label in train_set]
    dataset = TripletLabelDataset(train_data, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ArcFaceIrisNet(num_classes=len(class_to_idx), hash_bits=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    criterion = MultiTaskIrisLoss(
        lambda_hash=0.1,
        lambda_triplet=1.0,
        lambda_center=0.1,
        num_classes=len(class_to_idx),
        label_smoothing=0.1
    )

    train(model, dataloader, val_set, optimizer, criterion, device, transform,
          output_dir, class_to_idx, epochs=30)
