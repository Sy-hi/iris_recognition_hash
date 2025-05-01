# iris_recognition_hash
# Iris Recognition with Deep Hash Encoding

This project implements an end-to-end iris recognition system that integrates deep learning-based segmentation, feature extraction, hash encoding, and evaluation. It includes:

- **Iris segmentation** using a U-Net architecture.
- **Feature extraction and training** with an EfficientNet backbone and ArcFace loss.
- **Binary hash encoding** for efficient matching.
- **Graphical user interface (GUI)** for one-click recognition.
- **Evaluation scripts** for precision, recall, F1-score, and Hamming distance.

## Project Structure

- `seg_image.py` – Predict iris masks using the U-Net model.
- `train_arcface_model_strong_v3.py` – Train the ArcFace model with EfficientNet backbone.
- `tsne_visualizer.py` – Visualize feature distributions via t-SNE.
- `threshold_search.py` – Automatically search for the best Hamming threshold.
- `GUI.py` – Tkinter-based interface for image selection and result display.
- `unet_model.pth` – Pretrained segmentation model (not included due to size limit).
- `train_model_2.py` – Alternative training script with hash loss and center loss integration.

Note: Large files (e.g., `.pth`, `.csv`, `.zip`) have been excluded due to GitHub's 100MB file limit. Please contact the author for complete model files if needed.

## Requirements
torch
torchvision
tqdm
numpy
opencv-python
pillow
matplotlib
scikit-learn
efficientnet_pytorch

# ==================== ASSETS BY SCRIPT ====================

  ## seg.py
  - unet_model.pth
  
  ## train_arcface_model_strong_v3.py
  - log.csv
  - training_curve.png
  - match_log.csv
  - hamming_hist.png
  - train_encodings.csv
  - train_encodings.pkl
  - val_encodings.csv
  - val_encodings.pkl
  - threshold.txt
  - tsne_hash.png
  
  ## infer_identity_from_image.py
  - unet_model.pth
  - train_encodings.csv
  - threshold.txt
  - (test.png)
  	-input image
  - seg_mask.png
  - seg_overlay.png
  - normalized_half.png
  - encoded_input.png
  
  ## train_model.py
  - unet_model.pth
  
  ## save_encodings_and_analyze.py
  - hamming_stats.csv
  
  ## threshold_search.py
  - threshold_result.txt
  
  ## tsne_visualizer.py
  - tsne_hash.png


## Quick Start

```bash
# Segment iris from image to check seg_model actual performance
python seg_image.py --img_path example.png

# Train feature extractor and save encoding result
python train_arcface_model_strong_v3.py

# Launch GUI
python GUI.py
