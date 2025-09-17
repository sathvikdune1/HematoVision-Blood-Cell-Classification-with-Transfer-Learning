import os
import cv2

# Base directory
base_dir = "D:/apsche-project/dataset/images/JPEGImages"

# Sample logic to map your CSV/index image numbers to actual file names
def get_image_path(image_index):
    # You may need to adjust this function depending on your file naming pattern
    file_name = f"BloodImage_{int(image_index):05d}.jpg"  # Pads with 0s to match your filenames
    return os.path.join(base_dir, file_name)

# Example usage
image_indices = [0, 1, 2, 3, 4, 5]  # Replace with actual indices from your labels file or dataset
for idx in image_indices:
    img_path = get_image_path(idx)
    
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
    else:
        print(f"Loaded image: {img_path}, shape: {img.shape}")
