import tensorflow as tf
from tensorflow.keras import layers, models

def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=4):

    # ====== Load pretrained MobileNetV2 ======
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # ====== Freeze base model ======
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# ====== Build & print summary ======
model = build_mobilenet_model()
model.summary()
