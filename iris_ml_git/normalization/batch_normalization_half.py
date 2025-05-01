# Normalize iris and retain bottom half after unwrapping with mask
# Save lower half iris image

import os
import cv2
import numpy as np

def normalize_iris_with_mask(image, mask, radial_res=64, angular_res=512):
    # Find outermost contour from mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Mask not found")
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), iris_radius = cv2.minEnclosingCircle(contour)
    center = (int(cx), int(cy))
    iris_radius = int(iris_radius)

    # Generate polar coordinates grid
    theta = np.linspace(0, 2 * np.pi, angular_res)
    r = np.linspace(0, 1, radial_res)
    theta_grid, r_grid = np.meshgrid(theta, r)

    # Map polar coordinates to Cartesian
    x = center[0] + r_grid * iris_radius * np.cos(theta_grid)
    y = center[1] + r_grid * iris_radius * np.sin(theta_grid)

    # Perform remapping on image and mask
    normalized = cv2.remap(image, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    mask_normalized = cv2.remap(mask, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_NEAREST)
    normalized[mask_normalized == 0] = 0
    return normalized

def batch_normalize_images(image_root, mask_root, output_root, radial_res=64, angular_res=512):
    # Traverse all image folders and normalize them one by one
    for dirpath, _, filenames in os.walk(image_root):
        if not any(f.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg')) for f in filenames):
            continue  # Skip folders without valid images

        for filename in filenames:
            if not filename.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg')):
                continue

            image_path = os.path.join(dirpath, filename)
            mask_path = os.path.join(mask_root, filename)

            if not os.path.exists(mask_path):
                print(f"[Skip] Mask not found：{mask_path}")
                continue

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                print(f"[Skip] Failed to load image or mask：{image_path}")
                continue

            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            try:
                normalized = normalize_iris_with_mask(image, binary_mask, radial_res, angular_res)
            except Exception as e:
                print(f"[Error] Normalization failed：{image_path} - {e}")
                continue

            #Keep lower half of the normalized iris
            normalized = normalized[normalized.shape[0] // 2:, :]

            # Construct output path with folder structure preserved
            rel_path = os.path.relpath(image_path, image_root)
            output_path = os.path.join(output_root, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, normalized)
            print(f"[Saved] {output_path}")

if __name__ == "__main__":
    image_root = r"C:\Users\User\Desktop\unsw\y2\9773\ELEC9773 Databases\Database 1 - CASIA v3.0-Iris-Interval\after_process\Processed"
    mask_root = r"C:\Users\User\Desktop\unsw\y2\9773\ELEC9773 Databases\Database 1 - CASIA v3.0-Iris-Interval\after_process\Segmentation groundtruth"
    output_root = r"C:\Users\User\Desktop\unsw\y2\9773\ELEC9773 Databases\Database 1 - CASIA v3.0-Iris-Interval\after_process\Normalized_half"

    batch_normalize_images(image_root, mask_root, output_root)
