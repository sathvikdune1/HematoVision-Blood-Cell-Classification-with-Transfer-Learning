import os
import shutil
import pandas as pd

# Path to CSV and images
csv_path = 'dataset/labels_cleaned.csv'
image_dir = 'dataset/images/JPEGImages'
output_dir = 'dataset/prepared'

# Read the CSV file
df = pd.read_csv(csv_path)
print(f"Total entries in label file: {len(df)}")

# Create output folder
os.makedirs(output_dir, exist_ok=True)

missing_count = 0

# Process each row
for _, row in df.iterrows():
    image_id = int(row['image'])
    file = f"BloodImage_{image_id:05d}.jpg"
    label = row['Category']

    src_path = os.path.join(image_dir, file)
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    dst_path = os.path.join(label_dir, file)
    if os.path.exists(src_path):
        shutil.copyfile(src_path, dst_path)
    else:
        missing_count += 1

print(f"\nDataset preparation completed.")
print(f"Missing images skipped: {missing_count}")
