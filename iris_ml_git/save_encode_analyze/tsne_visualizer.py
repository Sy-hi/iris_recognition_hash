# Visualize t-SNE projection of hash codes from a trained iris recognition model


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image

def visualize_tsne(model, val_paths, transform, device, save_path="tsne_hash.png"):
    model.eval()  # set model to evaluation mode
    features, labels = [], []

    # Extract hash features from validation images
    with torch.no_grad():
        for path, label in val_paths:
            img = transform(Image.open(path).convert("L")).unsqueeze(0).to(device)  # preprocess image
            hash_code, _ = model(img, None)  # forward pass through model
            features.append(hash_code.cpu().numpy()[0])  # convert tensor to numpy
            labels.append(label)

    # Convert feature list to numpy array
    features = np.array(features)

    # Apply t-SNE to reduce high-dimensional hash codes to 2D
    tsne = TSNE(n_components=2, random_state=42).fit_transform(features)

    # Plot each label category as a separate color group
    plt.figure(figsize=(10, 8))
    for l in set(labels):
        idx = [i for i, v in enumerate(labels) if v == l]
        plt.scatter(tsne[idx, 0], tsne[idx, 1], label=str(l), s=5)

    # Format and save the plot
    plt.title("t-SNE of Hash Codes")
    plt.legend(fontsize=5, loc="upper right", ncol=2)
    plt.savefig(save_path, dpi=300)
    plt.close()
