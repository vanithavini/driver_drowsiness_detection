import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    layout="wide"
)
st.title("🚗 Driver Drowsiness Detection System")

# ====== LOAD MODEL ======
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/mobilenet_final.h5")
    return model
model = load_model()
classes = ['Closed','Open','no_yawn','yawn']
IMG_SIZE = 224

# ====== Best model metrics ======
BEST_MODEL = "MobileNetV2"
BEST_ACCURACY = 0.8899
BEST_LOSS = 0.2194

# ====== IMAGE PREPROCESS ======
def preprocess(img):
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img/255.0
    img = np.expand_dims(img,axis=0)
    return img

# ====== FATIGUE STAGE MAPPING ======
def fatigue_stage(label):
    if label == "Open" or label == "no_yawn":
        return "ALERT","green"
    elif label == "Closed":
        return "MILD FATIGUE","orange"
    else:
        return "SEVERE FATIGUE","red"

# ====== SIDEBAR NAVIGATION ======
page = st.sidebar.radio(
    "Navigation",
    ["📊 Model Evaluation",
     "🖼 Image Prediction",
     "🎥 Real Time Detection",
     "📈 Fatigue Progression"]
)

# ====== PAGE 1 — MODEL EVALUATION ======
if page == "📊 Model Evaluation":
    st.header("Model Evaluation Results")
    col1,col2,col3 = st.columns(3)
    col1.metric("Best Model", BEST_MODEL)
    col2.metric("Accuracy", f"{BEST_ACCURACY*100:.2f}%")
    col3.metric("Loss", f"{BEST_LOSS:.4f}")
    st.divider()
    st.subheader("Model Accuracy Comparison")
    st.image("outputs/accuracy_comparison.png")
    st.subheader("Model Loss Comparison")
    st.image("outputs/loss_comparison.png")
    st.subheader("Confusion Matrix (Best Model)")
    st.image("outputs/confusion_matrix.png")

# ====== PAGE 2 — IMAGE PREDICTION ======
elif page == "🖼 Image Prediction":
    st.header("Upload Image for Prediction")
    uploaded = st.file_uploader(
        "Upload driver image",
        type=["jpg","png","jpeg"]
    )
    if uploaded is not None:
        image = Image.open(uploaded)
        img = np.array(image)
        st.image(image,width=350)
        img = preprocess(img)
        pred = model.predict(img)[0]
        index = np.argmax(pred)
        label = classes[index]
        stage,color = fatigue_stage(label)
        st.subheader(f"Prediction : {label}")
        st.markdown(
            f"### Driver Status : :{color}[{stage}]"
        )
        prob_dict = {
            classes[i]: float(pred[i])
            for i in range(len(classes))
        }
        st.bar_chart(prob_dict)

# ====== PAGE 3 — REAL TIME DETECTION ======
elif page == "🎥 Real Time Detection":
    import time
    import winsound
    st.header("Live Driver Monitoring")
    start = st.button("Start Detection")
    FRAME_WINDOW = st.image([])
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if start:
        cap = cv2.VideoCapture(0)
        eye_closed_counter = 0
        yawn_counter = 0
        EYE_THRESHOLD = 12
        YAWN_THRESHOLD = 8
        prev_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Camera not working")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (224,224))
                face = face/255.0
                face = np.reshape(face, (1,224,224,3))
                prediction = model.predict(face, verbose=0)
                class_index = np.argmax(prediction)
                label = classes[class_index]
                confidence = np.max(prediction)
                text = f"{label} ({confidence:.2f})"

                # draw rectangle
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                # ===== Eye detection =====
                if label == "Closed":
                    eye_closed_counter += 1
                else:
                    eye_closed_counter = 0

                # ===== Yawn detection =====
                if label == "yawn":
                    yawn_counter += 1
                else:
                    yawn_counter = 0

                # ===== Decision Fusion =====
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

                # show fatigue state
                cv2.putText(frame, state, (50,70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # ===== FPS =====
            current_time = time.time()
            fps = 1/(current_time-prev_time) if prev_time!=0 else 0
            prev_time = current_time
            cv2.putText( frame, f"FPS:{int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            FRAME_WINDOW.image(frame, channels="BGR")
        cap.release()

# ====== PAGE 4 — FATIGUE PROGRESSION ======
elif page == "📈 Fatigue Progression":
    st.header("Driver Fatigue Progression Over Time")
    st.image("outputs/fatigue_curve.png")
    st.markdown(
    """
    ### Interpretation

    The fatigue curve shows how the driver's condition changes during a driving session.

    **Alert → Mild Fatigue → Severe Fatigue**

    This progression helps detect early warning signs of drowsiness and can be used
    to trigger driver alerts in real-world applications.
    """
    )

