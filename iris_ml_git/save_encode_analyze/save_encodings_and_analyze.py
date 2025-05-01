# Iris hash encoding, Hamming analysis, threshold optimization, and recognition accuracy evaluation

import os
import torch
import pickle
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt


# Extract person ID and eye side from filename
def extract_info(filename):
    pid = filename[:5]
    eye = filename[5].upper()
    return pid, eye


# Compute Hamming distance between two binary arrays
def hamming_distance(a, b):
    return (a != b).sum().item()


# Save hash encodings to both CSV and PKL formats
def save_encodings(model, image_paths, transform, device, csv_path, pkl_path):
    model.eval()
    encodings = {}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Get hash length from a dummy image
        with torch.no_grad():
            dummy_input = transform(Image.open(image_paths[0][0]).convert("L")).unsqueeze(0).to(device)
            dummy_hash, _ = model(dummy_input, None)
            hash_len = dummy_hash.shape[1]

        writer.writerow(["image", "person_id", "eye"] + [f"bit_{i}" for i in range(hash_len)])

        for path, label in tqdm(image_paths, desc="Encoding"):
            img_name = os.path.basename(path)
            pid, eye = extract_info(img_name)

            img = transform(Image.open(path).convert("L")).unsqueeze(0).to(device)
            with torch.no_grad():
                hash_code, _ = model(img, None)
                binary = (hash_code > 0).int().squeeze().cpu()

            encodings[img_name] = {"id": pid, "eye": eye, "hash": binary}
            writer.writerow([img_name, pid, eye] + binary.tolist())

    with open(pkl_path, "wb") as pf:
        pickle.dump(encodings, pf)

    return encodings


# Analyze Hamming distances and search for the best threshold based on F1 score
def compute_hamming_statistics(encodings, save_path="hamming_stats.csv", plot_path=None, thresh_txt_path=None):
    items = list(encodings.items())
    threshold_candidates = list(range(10, 201, 5))
    best_thresh, best_f1 = 0, 0
    intra, inter = [], []

    # Compute pairwise Hamming distances and save to CSV
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img1", "img2", "label", "hamming"])
        for i in range(len(items)):
            name1, info1 = items[i]
            for j in range(i + 1, len(items)):
                name2, info2 = items[j]
                same_person = info1["id"] == info2["id"]
                same_eye = info1["eye"] == info2["eye"]
                label = "intra" if same_person and same_eye else "inter"
                dist = hamming_distance(info1["hash"], info2["hash"])
                writer.writerow([name1, name2, label, dist])
                if label == "intra":
                    intra.append(dist)
                else:
                    inter.append(dist)

    # Evaluate F1 score at each threshold
    for thresh in threshold_candidates:
        tp = sum(d <= thresh for d in intra)
        fp = sum(d <= thresh for d in inter)
        fn = sum(d > thresh for d in intra)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Save threshold summary
    if thresh_txt_path:
        with open(thresh_txt_path, "w") as f:
            f.write(f"best_threshold={best_thresh}\n")
            f.write(f"intra_range={min(intra)} ~ {max(intra)}\n")
            f.write(f"inter_range={min(inter)} ~ {max(inter)}\n")

    # Plot histogram of Hamming distances
    if plot_path:
        plt.hist(intra, bins=40, alpha=0.7, label="Intra-personal")
        plt.hist(inter, bins=40, alpha=0.7, label="Inter-personal")
        plt.axvline(best_thresh, color="red", linestyle="--", label=f"Best Threshold={best_thresh}")
        plt.xlabel("Hamming Distance")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.title("Hamming Distance Distribution")
        plt.savefig(plot_path)
        plt.close()

    return best_thresh, (min(intra), max(intra)), (min(inter), max(inter))


# Evaluate recognition accuracy based on threshold
def recognition_accuracy(test_encodings, db_encodings, threshold):
    correct = 0
    total = 0

    for test_name, test_info in tqdm(test_encodings.items(), desc="Testing"):
        test_hash = test_info["hash"]
        true_id = test_info["id"]

        best_match, best_dist = None, 999
        for db_name, db_info in db_encodings.items():
            db_hash = db_info["hash"]
            dist = hamming_distance(test_hash, db_hash)
            if dist < best_dist:
                best_dist = dist
                best_match = db_info["id"]

        predicted = best_match if best_dist <= threshold else "unknown"
        if predicted == true_id:
            correct += 1
        total += 1

    return 100.0 * correct / total
