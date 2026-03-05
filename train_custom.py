import tensorflow as tf
from preprocessing import train_generator, val_generator
from custom_cnn_model import build_model

# Build model
model = build_model()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "custom_best.h5",
    monitor='val_accuracy',
    save_best_only=True
)

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save("custom_final.h5")

