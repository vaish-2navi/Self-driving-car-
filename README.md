# Self-driving-car-
# 🚗 Self-Driving Car Simulator

A Self-Driving Car Simulator project developed using the **Udacity Self-Driving Car Simulator**. This project demonstrates the process of collecting driving data, training a machine learning model, and testing autonomous driving behavior in a simulated environment.

---

## 📌 Project Overview

The goal of this project is to simulate an autonomous vehicle that learns to drive by imitating human driving behavior. The simulator records driving images and steering angles, which are later used to train a deep learning model.

---

## 🎯 Objectives

- Collect driving data using the Udacity Simulator.
- Record center camera images while driving.
- Train a machine learning model using the collected dataset.
- Test the trained model in autonomous mode.
- Understand the basics of Behavioral Cloning for Self-Driving Cars.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Pandas
- Udacity Self-Driving Car Simulator
- Git & GitHub

---

## 📂 Project Structure

```
Self-driving-car/
│
├── IMG/                  # Recorded driving images
├── driving_log.csv       # Driving data (steering, throttle, brake, speed)
├── model.py              # Model training script
├── drive.py              # Autonomous driving script
├── model.h5              # Trained model
├── README.md             # Project documentation
└── requirements.txt      # Required Python libraries
```

---

## ⚙️ How It Works

1. Launch the Udacity Self-Driving Car Simulator.
2. Select **Training Mode**.
3. Drive the car manually.
4. Click the **Record** button to collect driving data.
5. The simulator saves:
   - Driving images
   - Steering angle
   - Throttle
   - Brake
   - Speed
6. Train the deep learning model using the collected data.
7. Load the trained model.
8. Run the simulator in **Autonomous Mode**.

---

## 📊 Dataset

The dataset contains:

- Center camera images
- Steering angle
- Throttle
- Brake
- Speed

The images are stored in the **IMG** folder and the corresponding values are stored in **driving_log.csv**.

---

## ▶️ Running the Project

### Clone the Repository

```bash
git clone https://github.com/your-username/Self-driving-car.git
```

### Navigate to the Project

```bash
cd Self-driving-car
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python model.py
```

### Run Autonomous Driving

```bash
python drive.py model.h5
```

---

## 📸 Screenshots

Add screenshots of:

- Udacity Simulator
- Training Mode
- Autonomous Mode
- Recorded Images

---

## 📈 Future Improvements

- Improve steering angle prediction
- Add image preprocessing
- Data augmentation
- Better CNN architecture
- Lane detection integration
- Traffic sign recognition
- Object detection
- Real-time performance optimization

---

## 📚 Learning Outcomes

Through this project, I learned:

- Behavioral Cloning
- Data Collection
- Image Processing
- Deep Learning Basics
- CNN for Autonomous Driving
- Git and GitHub
- Project Documentation

---

## 🙏 Acknowledgements

This project was developed for learning purposes using the **Udacity Self-Driving Car Simulator**.

Special thanks to:
- Udacity
- TensorFlow
- Keras
- OpenCV Community

---

