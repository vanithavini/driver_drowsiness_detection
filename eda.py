import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "dataset/train"
CLASSES = ["yawn", "no_yawn", "Open", "Closed"] 

# ====== 1️⃣ Class Distribution ======
class_counts = {}

for cls in CLASSES:
    class_path = os.path.join(DATA_DIR, cls)
    class_counts[cls] = len(os.listdir(class_path))

plt.figure()
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
plt.title("Class Distribution (Training Set)")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.show()

print("Class Counts:", class_counts) 

# ====== 2️⃣ Show Sample Images ======
plt.figure(figsize=(10, 8))

for i, cls in enumerate(CLASSES):
    class_path = os.path.join(DATA_DIR, cls)
    img_name = os.listdir(class_path)[0]
    img_path = os.path.join(class_path, img_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.tight_layout()
plt.show() 

# ====== 3️⃣ Image Dimension Analysis ======
heights = []
widths = []

for cls in CLASSES:
    class_path = os.path.join(DATA_DIR, cls)
    for img_name in os.listdir(class_path)[:50]:  # sample 50 per class
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        heights.append(h)
        widths.append(w)

print("Average Height:", np.mean(heights))
print("Average Width:", np.mean(widths))


