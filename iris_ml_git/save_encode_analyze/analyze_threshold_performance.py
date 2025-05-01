# Evaluate performance at multiple thresholds and find the best accuracy, F1 score, and EER

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


# Load encodings and labels from a pickle file
def load_encodings(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


# Extract person ID from image filename
def get_label(image_name):
    return image_name[:5]  # e.g., S1001L01 → S1001


# Compute Hamming distance between two binary codes
def compute_hamming_dist(a, b):
    return np.sum(a != b)


# Evaluate accuracy, F1 score, FAR, FRR at different thresholds and plot results
def analyze_threshold_performance(val_enc, train_enc, output_dir, hash_bits=256):
    print("[1/4] Computing Hamming distances...")

    thresholds = list(range(0, hash_bits + 1))
    correct = []
    FAR_list = []
    FRR_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    for threshold in tqdm(thresholds, desc="[2/4] Evaluating thresholds"):
        hits = total = false_accepts = false_rejects = 0
        preds = []
        gts = []

        # For each validation code, find the best matching training code
        for val in val_enc:
            v_code = np.array(val['code'])
            v_label = get_label(val['image'])

            best_match = None
            best_dist = float('inf')

            for tr in train_enc:
                tr_code = np.array(tr['code'])
                tr_label = get_label(tr['image'])

                dist = compute_hamming_dist(v_code, tr_code)
                if dist < best_dist:
                    best_dist = dist
                    best_match = tr_label

            norm_dist = best_dist / hash_bits
            match = best_match if best_dist <= threshold else "unknown"

            preds.append(match)
            gts.append(v_label)

            if match == v_label:
                hits += 1
            if match != "unknown" and match != v_label:
                false_accepts += 1
            if match == "unknown" and v_label in [get_label(t['image']) for t in train_enc]:
                false_rejects += 1

            total += 1

        acc = hits / total
        correct.append(acc)
        FAR_list.append(false_accepts / total)
        FRR_list.append(false_rejects / total)

        # Compute classification metrics
        preds_np = np.array(preds)
        gts_np = np.array(gts)
        tp = np.sum((preds_np == gts_np) & (preds_np != 'unknown'))
        fp = np.sum((preds_np != gts_np) & (preds_np != 'unknown'))
        fn = np.sum((preds_np == 'unknown') & (gts_np != 'unknown'))

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # Determine best accuracy threshold
    best_idx = int(np.argmax(correct))
    best_thresh = thresholds[best_idx]
    best_acc = correct[best_idx]

    # Determine best F1 score threshold
    best_f1_idx = int(np.argmax(f1_list))
    best_f1_thresh = thresholds[best_f1_idx]
    best_f1_score = f1_list[best_f1_idx]

    # Determine Equal Error Rate (EER) threshold
    diffs = np.abs(np.array(FAR_list) - np.array(FRR_list))
    eer_idx = np.argmin(diffs)
    eer_thresh = thresholds[eer_idx]
    eer_val = (FAR_list[eer_idx] + FRR_list[eer_idx]) / 2

    print(
        f"\n[3/4] Best Accuracy: {best_acc:.4f} at threshold {best_thresh} (normalized: {best_thresh / hash_bits:.4f})")
    print(
        f"[3/4] Best F1 Score: {best_f1_score:.4f} at threshold {best_f1_thresh} (normalized: {best_f1_thresh / hash_bits:.4f})")
    print(f"[3/4] EER: {eer_val:.4f} at threshold {eer_thresh} (normalized: {eer_thresh / hash_bits:.4f})")

    # Plot accuracy vs threshold
    plt.figure()
    plt.plot(np.array(thresholds) / hash_bits, correct, label="Accuracy")
    plt.axvline(best_thresh / hash_bits, color='green', linestyle='--', label=f"Best Acc ({best_thresh})")
    plt.xlabel("Threshold (Normalized)")
    plt.ylabel("Accuracy")
    plt.title("Threshold vs Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "threshold_vs_accuracy.png"))

    # Plot FAR and FRR vs threshold
    plt.figure()
    plt.plot(np.array(thresholds) / hash_bits, FAR_list, label="FAR", color='red')
    plt.plot(np.array(thresholds) / hash_bits, FRR_list, label="FRR", color='blue')
    plt.axvline(eer_thresh / hash_bits, color='purple', linestyle='--', label=f"EER ({eer_thresh})")
    plt.xlabel("Threshold (Normalized)")
    plt.ylabel("Rate")
    plt.title("FAR and FRR vs Threshold")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "far_frr_vs_threshold.png"))

    # Plot F1 score vs threshold
    plt.figure()
    plt.plot(np.array(thresholds) / hash_bits, f1_list, label="F1 Score", color="orange")
    plt.axvline(best_f1_thresh / hash_bits, color="green", linestyle="--", label=f"Best F1 ({best_f1_thresh})")
    plt.xlabel("Threshold (Normalized)")
    plt.ylabel("F1 Score")
    plt.title("Threshold vs F1 Score")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "threshold_vs_f1.png"))

    return best_thresh, eer_thresh, best_acc, best_f1_thresh, best_f1_score
