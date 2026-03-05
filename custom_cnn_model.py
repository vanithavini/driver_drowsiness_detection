import tensorflow as tf
from tensorflow.keras import layers, models

# ====== BUILD CNN MODEL ======
def build_model(input_shape=(224, 224, 3), num_classes=4):

    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2,2))

    # Block 2
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    # Block 3
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    # Flatten
    model.add(layers.Flatten())

    # Dense Layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# ====== CREATE & COMPILE MODEL ======
model = build_model()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ====== PRINT MODEL SUMMARY ======
model.summary()

