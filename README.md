# Self-Driving Car Simulator

An end-to-end behavioral-cloning project that uses a convolutional neural network (CNN) to predict steering angles from simulator camera images. The trained model can connect to the included Udacity beta simulator and drive in autonomous mode.

## Features

- NVIDIA-inspired CNN implemented with TensorFlow/Keras
- Image preprocessing: RGB conversion, cropping, resizing, and normalization
- Training augmentation with horizontal flips and brightness variation
- WebSocket driving server for the simulator
- Included pretrained model (`model.keras`) for a quick start

## Project structure

```text
Self-driving-car-/
├── drive.py           # Starts the autonomous-driving server
├── train.py           # Trains a model from recorded driving data
├── model.py           # CNN architecture
├── utils.py           # Dataset loading, preprocessing, and augmentation
├── model.keras        # Included trained model
├── requirements.txt   # Python dependencies
└── training_loss.png  # Training/validation loss plot
```

The Windows simulator is located one folder above this project:

```text
../beta_simulator.exe
```

## Requirements

- Windows (the included simulator build is a Windows executable)
- Python 3
- pip
- A working graphics driver capable of running the Unity simulator

## Installation

1. Clone the repository and enter the project folder.

   ```powershell
   git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   cd YOUR-REPOSITORY/Self-driving-car-
   ```

   If you downloaded the project instead, open PowerShell in the `Self-driving-car-` folder.

2. Create and activate a virtual environment.

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run the following once in the current terminal and activate again:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

3. Install the dependencies.

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Run the pretrained autonomous car

1. Confirm that `model.keras` is present in the `Self-driving-car-` folder.

2. Start the driving server from that folder.

   ```powershell
   python drive.py
   ```

   Wait until the terminal displays `Listening on port 4567...`.

3. In a separate window, launch the simulator.

   ```powershell
   ..\beta_simulator.exe
   ```

4. In the simulator, select a track, choose **Autonomous Mode**, and start the run. The simulator connects to the Python server at port `4567` and sends camera frames to the model.

5. Keep the terminal open while driving. It shows the current speed, predicted steering angle, and throttle value. Use the simulator controls to stop or reset the car when needed.

## Train your own model

Training data is not included. Record driving data in the simulator and arrange it as follows:

```text
