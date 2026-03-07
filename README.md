# 🚗 Driver Drowsiness Detection using Eye Closure and Yawning Analysis with Deep Learning.

## 📌 Project Overview
Driver drowsiness is one of the major causes of road accidents worldwide. When drivers become fatigued, their reaction time slows down, increasing the risk of accidents.
This project presents an **AI-based Driver Drowsiness Detection System** that uses **Deep Learning and Computer Vision** to monitor the driver's face and detect signs of fatigue such as **eye closure and yawning**.

The system can:
- Detect **eye closure**
- Detect **yawning**
- Classify driver state
- Provide **real-time fatigue alerts**
- Visualize **fatigue progression over time**
- Provide a **Streamlit web interface**

The system uses **CNN and Transfer Learning models (MobileNetV2 and EfficientNetB0)** to achieve high accuracy and real-time performance.

# 🎯 Objectives
The main objectives of this project are:
- Detect driver drowsiness using **deep learning models**
- Compare different models for best performance
- Implement **real-time fatigue detection using webcam**
- Build a **user-friendly Streamlit application**
- Visualize **driver fatigue progression**

# 🧠 Technologies Used
### Programming Language
- Python
### Deep Learning
- TensorFlow
- Keras
### Computer Vision
- OpenCV
### Data Processing
- NumPy
- Pandas
### Visualization
- Matplotlib
- Seaborn
### Deployment
- Streamlit

# 📂 Project Structure
Driver-Drowsiness-Detection/
│
├── dataset/
│ ├── train/
│ ├── val/
│ └── test/
│
├── models/
│ ├── custom_best.h5
│ ├── custom_final.h5
│ ├── mobilenet_best.h5
│ ├── mobilenet_final.h5
│ └── efficientnet_model.keras
│
├── outputs/
│ ├── accuracy_comparison.png
│ ├── loss_comparison.png
│ ├── confusion_matrix.png
│ └── fatigue_curve.png
│
├── src/
│ ├── module1_dataset_split.py
│ ├── module2_eda.py
│ ├── module3_preprocessing.py
│ ├── module4_custom_cnn_model.py
│ ├── module5_mobilenet_model.py
│ ├── module6_efficientnet_model.py
│ ├── module7A_train_custom.py
│ ├── module7B_train_mobilenet.py
│ ├── module7c_train_efficient.py
│ ├── module8_evaluate_models.py
│ ├── module9_realtime_detection.py
│ └── module10_fatigue_progression_curve.py
│
├── app.py
├── requirements.txt
└── README.md

# ⚙️ Workflow of the Project
The system follows a complete **machine learning pipeline**.
### 1️⃣ Dataset Preparation
- Dataset split into:
  - Train
  - Validation
  - Test

### 2️⃣ Exploratory Data Analysis
- Class distribution analysis
- Sample image visualization
- Image dimension analysis
- Pixel intensity distribution

### 3️⃣ Data Preprocessing
- Image resizing
- Normalization
- Data augmentation

### 4️⃣ Model Development
Three models were implemented:
| Model | Description |
|------|-------------|
| Custom CNN | Basic CNN architecture |
| MobileNetV2 | Lightweight transfer learning model |
| EfficientNetB0 | Advanced transfer learning model |

### 5️⃣ Model Training
Models were trained using:
- Adam optimizer
- Categorical cross entropy
- Early stopping
- Model checkpoints

### 6️⃣ Model Evaluation
Models were evaluated using:
- Accuracy
- Loss
- Confusion Matrix
- Classification Report

The **best performing model was MobileNetV2**.

# 📊 Model Performance
| Model | Accuracy | Loss |
|------|---------|------|
| Custom CNN | ~86% | Moderate |
| MobileNetV2 | **~89%** | Best |
| EfficientNetB0 | ~88% | Slightly higher |

MobileNetV2 was selected for **real-time deployment** due to its **speed and efficiency**.

# 🎥 Real-Time Driver Monitoring
The system uses:
- **Webcam input**
- **Face detection (Haar Cascade)**
- **Deep learning classification**
Driver states detected:
| Class | Meaning |
|------|--------|
| Open | Eyes open |
| Closed | Eyes closed |
| no_yawn | No yawning |
| yawn | Yawning |

### Driver Fatigue States
| State | Condition |
|------|-----------|
| Alert | Normal driver |
| Mild Fatigue | Yawning detected |
| Severe Fatigue | Eyes closed repeatedly |

An **audio alert** is triggered during severe fatigue.

# 📈 Fatigue Progression Analysis
The system also analyzes **fatigue progression over time**.
A fatigue curve shows:
Alert → Mild Fatigue → Severe Fatigue
This helps analyze **driver behavior during long driving sessions**.

# 🌐 Streamlit Web Application
A user-friendly web interface was built using **Streamlit**.
### Features
#### 1️⃣ Model Evaluation Dashboard
Displays:
- Model accuracy
- Loss
- Confusion matrix
- Performance comparison
#### 2️⃣ Image Prediction
Users can upload an image and detect driver fatigue.
#### 3️⃣ Real-Time Detection
Webcam-based driver monitoring.
#### 4️⃣ Fatigue Progression Visualization
Displays fatigue trend over time.

# ▶️ How to Run the Project
### Step 1 — Clone the Repository
### Step 2 — Install Dependencies
    pip install -r requirements.txt
### Step 3 — Run Streamlit App
    streamlit run app.py

# 📸 Sample Output
### Real-Time Detection
- Face detection
- Driver state prediction
- Fatigue alert

### Visualization Outputs
- Accuracy comparison chart
- Loss comparison chart
- Confusion matrix
- Fatigue progression curve

# 🚀 Future Scope
Although the current system performs well, several improvements can be made in the future:
### 1️⃣ Eye Aspect Ratio (EAR) Integration
Use facial landmark detection to improve **eye closure detection accuracy**.
### 2️⃣ Mobile Application Integration
Deploy the system as a **mobile app for drivers**.
### 3️⃣ Edge Device Deployment
Deploy the system on **Raspberry Pi or embedded systems** for real vehicle integration.
### 4️⃣ Infrared Camera Support
Use **infrared cameras** for night-time detection.
### 5️⃣ Multi-driver Monitoring
Extend the system to monitor **multiple drivers in fleet vehicles**.
### 6️⃣ Integration with Vehicle Systems
Connect with:
- car alarms
- vibration alerts
- autonomous braking systems.

# 🏁 Conclusion
This project successfully demonstrates an **AI-based Driver Drowsiness Detection System** using **deep learning and computer vision techniques**.
The system was able to:
- Detect signs of driver fatigue such as **eye closure and yawning**
- Achieve high classification performance using **MobileNetV2**
- Perform **real-time driver monitoring using webcam**
- Provide **visual insights into fatigue progression**
- Deliver a **user-friendly Streamlit application**
Such intelligent systems can significantly **reduce road accidents caused by driver fatigue** and contribute to safer transportation systems.
This project shows how **AI can be applied to real-world safety problems** and serves as a strong foundation for future **smart driver assistance systems**.
