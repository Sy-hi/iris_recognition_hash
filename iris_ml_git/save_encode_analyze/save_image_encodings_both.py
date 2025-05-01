# Save hash encodings and labels to CSV and PKL formats

import torch
import csv
import pickle
from tqdm import tqdm
from PIL import Image

def save_image_encodings(model, image_paths, transform, device, csv_path, pkl_path):
    model.eval()
    all_encodings = [] # all binary encodings
    all_labels = [] # all person labels
    hash_len = model.hash_dim # length of hash code

    # Write to CSV
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["image", "person_id", "eye"] + [f"bit_{i}" for i in range(hash_len)])

        for i, (path, label) in enumerate(tqdm(image_paths, desc="Saving encodings")):
            img = transform(Image.open(path).convert("L")).unsqueeze(0).to(device)
            with torch.no_grad():
                hash_code, _ = model(img, None)

            binary = (hash_code[0] > 0).int().tolist()
            all_encodings.append(binary)
            all_labels.append(label)

            # Extract person ID from label
            person = label.split("_")[0] if "_" in label else label

            # Determine eye side based on folder structure
            if "/L/" in path or "\\L\\" in path:
                eye = "L"
            elif "/R/" in path or "\\R\\" in path:
                eye = "R"
            else:
                eye = "?"

            writer.writerow([path, person, eye] + binary)

    #Save as pickle
    with open(pkl_path, "wb") as f_pkl:
        pickle.dump({"encodings": all_encodings, "labels": all_labels}, f_pkl)

    # Return encoding results
    return {
        "encodings": all_encodings,
        "labels": all_labels
    }

