# Search the best Hamming threshold for binary feature matching using F1 score

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import os
from PIL import Image

def search_best_threshold(model, val_paths, transform, device, output_dir, save_name="threshold_result.txt"):
    model.eval()
    embeddings, labels = [], []

    # Generate binary embeddings for all validation images
    with torch.no_grad():
        for path, label in val_paths:
            img = transform(Image.open(path).convert("L")).unsqueeze(0).to(device)
            hash_code, _ = model(img, None)
            embeddings.append((hash_code > 0).int().cpu().numpy()[0])
            labels.append(label)

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    best_thresh = None
    best_f1 = 0

    # Evaluate thresholds from 5 to 69
    with open(os.path.join(output_dir, save_name), "w") as f:
        for thresh in range(5, 70, 1):
            y_true, y_pred = [], []
            for i in range(len(embeddings)):
                for j in range(len(embeddings)):
                    if i == j:
                        continue
                    ham_dist = np.sum(np.bitwise_xor(embeddings[i], embeddings[j]))
                    same = labels[i] == labels[j]
                    match = ham_dist <= thresh
                    y_true.append(int(same))
                    y_pred.append(int(match))

            # Compute macro F1, precision, recall
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

            f.write(f"thresh={thresh}, f1={f1:.4f}, prec={prec:.4f}, recall={recall:.4f}\n")
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

    print(f"Best threshold: {best_thresh} with F1={best_f1:.4f}")
