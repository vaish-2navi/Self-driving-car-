import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


def build_model(input_shape=(66, 200, 3), learning_rate=1e-4):
    """
    Builds and compiles the NVIDIA End-to-End Deep Learning Architecture
    for self-driving car behavioral cloning.

    Parameters:
        input_shape (tuple): Shape of preprocessed input images (height, width, channels).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tf.keras.Model: Compiled Keras model with Adam optimizer and Mean Squared Error loss.
    """
    model = Sequential([
        # 5 Convolutional Layers (NVIDIA Architecture)
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        
        # Dropout layer to mitigate overfitting
        Dropout(0.5),
        
        # Fully Connected Layers
        Flatten(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)  # Output: Steering angle prediction
    ])

    # Compile model using Adam optimizer and Mean Squared Error (MSE) loss
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model
