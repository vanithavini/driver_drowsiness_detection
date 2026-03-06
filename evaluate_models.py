import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# ====== Paths ======
DATASET_PATH = "dataset"
MODEL_PATH = "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ====== Test Data Generator ======
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())

print("Classes:", class_names)

# ====== Load Models ======
print("\nLoading models...")

custom_model = tf.keras.models.load_model("models/custom_final.h5")
mobilenet_model = tf.keras.models.load_model("models/mobilenet_final.h5")
efficient_model = tf.keras.models.load_model("models/efficientnet_model.keras")

models = {
    "Custom CNN": custom_model,
    "MobileNetV2": mobilenet_model,
    "EfficientNetB0": efficient_model
}

# ====== Evaluate Models ======
results = []

print("\nEvaluating models on test dataset...\n")

for name, model in models.items():
    loss, acc = model.evaluate(test_generator, verbose=1)

    print(f"{name} -> Accuracy: {acc:.4f}  Loss: {loss:.4f}")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Loss": loss
    })

# ====== Create Results Table ======
results_df = pd.DataFrame(results)

print("\nModel Comparison Table:")
print(results_df)

# ====== Accuracy Comparison Graph ======
plt.figure()
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.savefig("outputs/accuracy_comparison.png")
plt.show()

# ====== Loss Comparison Graph ======
plt.figure()
plt.bar(results_df["Model"], results_df["Loss"])
plt.title("Model Loss Comparison")
plt.ylabel("Loss")
plt.xlabel("Models")
plt.savefig("outputs/loss_comparison.png")
plt.show()

# ====== Select Best Model ======
best_model_name = results_df.loc[results_df["Accuracy"].idxmax()]["Model"]

print("\nBest Model:", best_model_name)

best_model = models[best_model_name]

# ====== Predictions for Confusion Matrix ======
print("\nGenerating predictions...")

predictions = best_model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# ====== Confusion Matrix ======
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names)

plt.title(f"Confusion Matrix ({best_model_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

# ====== Classification Report ======
print("\nClassification Report:\n")

report = classification_report(y_true, y_pred, target_names=class_names)

print(report)

