import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained best model
model = load_model("models/mobilenet_final.h5")

# Class labels (same order as training)
classes = ['Closed', 'Open', 'no_yawn', 'yawn']

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting Driver Drowsiness Detection...")

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

        # Resize same as training
        face = cv2.resize(face, (224, 224))

        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        prediction = model.predict(face, verbose=0)

        class_index = np.argmax(prediction)

        label = classes[class_index]

        confidence = np.max(prediction)

        text = f"{label} ({confidence:.2f})"

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(frame, text,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

        # Drowsiness condition
        if label == "Closed" or label == "yawn":

            cv2.putText(frame,
                        "DROWSINESS ALERT!",
                        (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

    cv2.imshow("Driver Drowsiness Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

