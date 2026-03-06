import cv2
import numpy as np
import time
import winsound
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/mobilenet_final.h5")

# Class labels
classes = ['Closed', 'Open', 'no_yawn', 'yawn']

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

# Counters for fatigue detection
eye_closed_counter = 0
yawn_counter = 0

# Threshold values
EYE_THRESHOLD = 12
YAWN_THRESHOLD = 8

# FPS calculation
prev_time = 0

print("Press Q to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        prediction = model.predict(face, verbose=0)

        class_index = np.argmax(prediction)
        label = classes[class_index]
        confidence = np.max(prediction)

        text = f"{label} ({confidence:.2f})"

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        # -------- Fatigue Logic -------- #

        # Eye detection
        if label == "Closed":
            eye_closed_counter += 1
        else:
            eye_closed_counter = 0

        # Yawn detection
        if label == "yawn":
            yawn_counter += 1
        else:
            yawn_counter = 0

        # -------- Decision Fusion -------- #

        if eye_closed_counter >= EYE_THRESHOLD:

            state = "Severe Fatigue"
            color = (0,0,255)

            winsound.Beep(1500, 800)

        elif yawn_counter >= YAWN_THRESHOLD:

            state = "Mild Fatigue"
            color = (0,165,255)

        else:

            state = "Alert"
            color = (0,255,0)

        # Display fatigue state
        cv2.putText(
            frame,
            state,
            (50,70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

    # -------- FPS Calculation -------- #

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    cv2.imshow("Driver Drowsiness Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
