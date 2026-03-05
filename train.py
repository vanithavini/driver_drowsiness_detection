import tensorflow as tf
from preprocessing import train_generator, val_generator
from model import build_model

# ====== BUILD MODEL ======
model = build_model()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ====== CALLBACKS ======
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.h5",
    monitor='val_accuracy',
    save_best_only=True
)

# ====== TRAIN MODEL ======
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=[early_stop, checkpoint]
)

# SAVE FINAL MODEL
model.save("final_model.h5")

