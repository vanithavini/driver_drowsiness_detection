import tensorflow as tf
from tensorflow.keras import layers, models

def build_efficientnet_model(input_shape=(224, 224, 3), num_classes=4):

    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

model = build_efficientnet_model()
model.summary() 
