import os
import shutil
import random
from sklearn.model_selection import train_test_split

# ====== PATHS ======
SOURCE_DIR = r"C:\Users\Vanitha\Downloads\archive (2)\train"
BASE_DIR = r"dataset"

CLASSES = ["yawn", "no_yawn", "Open", "Closed"]

# ====== CREATE FOLDER STRUCTURE ======
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(BASE_DIR, split, cls), exist_ok=True)

print("Folder structure created successfully.")

# ====== SPLITTING LOGIC ======
for cls in CLASSES:
    class_path = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(class_path)

    train_imgs, temp_imgs = train_test_split(images, test_size=0.30, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)

    # Copy files
    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(BASE_DIR, "train", cls, img))

    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(BASE_DIR, "val", cls, img))

    for img in test_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(BASE_DIR, "test", cls, img))

    print(f"{cls} → Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

print("\nDataset successfully split into train, val, test.")

