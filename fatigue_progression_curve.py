import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random

# ====== Load trained model ======
model = load_model("models/mobilenet_final.h5")
classes = ['Closed', 'Open', 'no_yawn', 'yawn']

# ====== Dataset path (TEST DATA) ======
DATASET_PATH = "dataset/test"
image_paths = []

for class_name in classes:
    class_folder = os.path.join(DATASET_PATH, class_name)
    for img in os.listdir(class_folder):
        image_paths.append(os.path.join(class_folder, img))
print("Total images found:", len(image_paths))

# ====== Randomly select 320 frames ======
open_imgs = []
yawn_imgs = []
closed_imgs = []

for img in image_paths:

    if "Open" in img or "no_yawn" in img:
        open_imgs.append(img)

    elif "yawn" in img:
        yawn_imgs.append(img)

    elif "Closed" in img:
        closed_imgs.append(img)

random.shuffle(open_imgs)
random.shuffle(yawn_imgs)
random.shuffle(closed_imgs)

frames = []

frames += open_imgs[:100]
frames += yawn_imgs[:100]
frames += closed_imgs[:100]

# ====== Prediction loop ======
fatigue_levels = []

for img_path in frames:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0    
    img = np.reshape(img, (1,224,224,3))
    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    label = classes[class_index]

    # ====== Mapping ======
    if label in ["Open","no_yawn"]:
        fatigue_levels.append(0)   # Alert
    elif label == "yawn":
        fatigue_levels.append(1)   # Mild fatigue
    elif label == "Closed":
        fatigue_levels.append(2)   # Severe fatigue

# ====== Convert frames to time intervals ======
frames_per_minute = 32
minutes = len(fatigue_levels) // frames_per_minute
fatigue_progression = []
for i in range(minutes):
    start = i * frames_per_minute
    end = start + frames_per_minute
    interval = fatigue_levels[start:end]
    fatigue_progression.append(np.mean(interval))

# ====== Plot progression curve ======
time_axis = list(range(1, minutes+1))
plt.figure()
plt.plot(time_axis, fatigue_progression, marker='o')
plt.yticks([0,1,2], ["Alert","Mild Fatigue","Severe Fatigue"])
plt.xlabel("Time (Minutes)")
plt.ylabel("Driver Fatigue Level")
plt.title("Driver Fatigue Progression Curve")
plt.grid()
plt.savefig("outputs/fatigue_curve.png")
plt.show()

# ====== Transition Detection ======
print("\nFatigue progression over time:\n")
for i, level in enumerate(fatigue_progression):
    if level < 0.5:
        state = "Alert"
    elif level < 1.5:
        state = "Mild Fatigue"
    else:
        state = "Severe Fatigue"
    print(f"Minute {i+1} : {state}")

