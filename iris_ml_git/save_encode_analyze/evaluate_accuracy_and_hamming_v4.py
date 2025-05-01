# Evaluate iris recognition accuracy and visualize Hamming distance distribution

import torch
import os
import csv
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

def extract_info_from_path(path):
    # Extract person ID and eye side from image filename
    fname = os.path.basename(path)
    person_id = fname[:5]  # e.g. S1001
    eye = fname[5].upper()  # L or R
    return person_id, eye

def hamming_distance(a, b):
    # Compute Hamming distance between two binary codes
    return (a != b).sum().item()

def evaluate_accuracy_and_hamming(model, val_paths, transform, device,
                                  save_csv, match_csv, hamming_threshold=25, save_fig=None):
    # Perform 1:N matching and evaluate recognition accuracy
    model.eval()
    results = []
    all_train = []

    # Build simulated training DB from validation paths
    for path, label in val_paths:
        pid, eye = extract_info_from_path(path)
        img = transform(Image.open(path).convert("L")).unsqueeze(0).to(device)
        with torch.no_grad():
            hash_code, _ = model(img, None)
            binary = (hash_code > 0).int().squeeze().cpu()
        all_train.append((pid, eye, binary, path))

    correct = 0
    intra_dists = [] # Distances for same person and same eye
    inter_dists = [] # Distances for different person and eye

    for path, label in tqdm(val_paths, desc="1:N Matching"):
        pid_q, eye_q = extract_info_from_path(path)
        img = transform(Image.open(path).convert("L")).unsqueeze(0).to(device)
        with torch.no_grad():
            query_code, _ = model(img, None)
            query_bin = (query_code > 0).int().squeeze().cpu()

        best_dist = 999
        best_id = None

        for pid_db, eye_db, db_bin, db_path in all_train:
            dist = hamming_distance(query_bin, db_bin)

            if pid_q == pid_db and eye_q == eye_db and path != db_path:
                intra_dists.append(dist)
            elif pid_q != pid_db:
                inter_dists.append(dist)

            if dist < best_dist:
                best_dist = dist
                best_id = pid_db

        pred_id = best_id if best_dist <= hamming_threshold else "unknown"
        results.append([os.path.basename(path), pid_q, pred_id, best_dist])

        if pred_id == pid_q:
            correct += 1

    acc = 100.0 * correct / len(val_paths)

    with open(save_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["val_image", "true_id", "predicted_id", "hamming_distance"])
        writer.writerows(results)

    # Save match stats for histogram
    if save_fig:
        plt.hist(intra_dists, bins=40, alpha=0.7, label="Intra-personal")
        plt.hist(inter_dists, bins=40, alpha=0.7, label="Inter-personal")
        plt.axvline(hamming_threshold, color="red", linestyle="--", label=f"Threshold={hamming_threshold}")
        plt.xlabel("Hamming Distance")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.title("Hamming Distance Distribution")
        plt.savefig(save_fig)
        plt.close()

    return acc
