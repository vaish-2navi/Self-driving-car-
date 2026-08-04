import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Image processing constants
CROP_TOP = 60
CROP_BOTTOM = 25
INPUT_HEIGHT = 66
INPUT_WIDTH = 200
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, 3)


def load_driving_data(data_path):
    """
    Loads driving_log.csv from data_path and reads ONLY the center image path
    and steering angle columns, ignoring left and right camera columns.

    Parameters:
        data_path (str): Path to directory containing driving_log.csv or direct path to CSV file.

    Returns:
        pd.DataFrame: DataFrame containing ONLY 'center' and 'steering' columns.
    """
    if os.path.isdir(data_path):
        csv_path = os.path.join(data_path, 'driving_log.csv')
    else:
        csv_path = data_path

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Driving log CSV file not found at: {csv_path}")

    # Load CSV file (handles CSV with or without header line)
    df = pd.read_csv(csv_path)
    
    # Check if header exists or standard column names are present
    if 'center' not in df.columns or 'steering' not in df.columns:
        df = pd.read_csv(csv_path, names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])

    # Filter to ONLY center camera image path and steering angle
    df = df[['center', 'steering']].copy()
    df['center'] = df['center'].astype(str).str.strip()
    df['steering'] = df['steering'].astype(float)

    return df


def resolve_image_path(image_path, data_dir):
    """
    Resolves image file path across operating systems and directory configurations.

    Parameters:
        image_path (str): Raw image path string from driving_log.csv.
        data_dir (str): Base data directory.

    Returns:
        str: Absolute or relative resolved filepath that exists on disk.
    """
    clean_path = image_path.replace('\\', '/')
    filename = os.path.basename(clean_path)

    candidate_paths = [
        image_path,
        os.path.join(data_dir, clean_path),
        os.path.join(data_dir, 'IMG', filename),
        os.path.join(data_dir, filename)
    ]

    for cand in candidate_paths:
        if os.path.exists(cand):
            return cand

    return image_path


def preprocess_image(image):
    """
    Preprocesses raw image loaded via OpenCV:
    1. Converts BGR color space to RGB.
    2. Crops non-informative regions (top sky, bottom car hood).
    3. Resizes image to NVIDIA model input shape (200x66).
    4. Normalizes pixel intensity values to range [0.0, 1.0].

    Parameters:
        image (np.ndarray): BGR image matrix.

    Returns:
        np.ndarray: Preprocessed RGB image matrix normalized to [0, 1].
    """
    # Convert BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crop sky (top 60px) and hood (bottom 25px)
    height = image.shape[0]
    image = image[CROP_TOP:height - CROP_BOTTOM, :, :]

    # Resize to (200, 66)
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)

    # Normalize pixel values
    image = image.astype(np.float32) / 255.0

    return image


def augment_image(image, steering_angle):
    """
    Applies data augmentation to training images:
    - Horizontal flipping with inverted steering angle.
    - Random brightness adjustment in HSV space.

    Parameters:
        image (np.ndarray): Preprocessed RGB image.
        steering_angle (float): Steering angle label.

    Returns:
        tuple: (augmented_image, adjusted_steering_angle)
    """
    aug_img = image.copy()
    aug_steering = steering_angle

    # 1. Random Horizontal Flip
    if np.random.rand() < 0.5:
        aug_img = cv2.flip(aug_img, 1)
        aug_steering = -aug_steering

    # 2. Random Brightness Adjustment
    if np.random.rand() < 0.5:
        img_uint8 = (aug_img * 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        
        # Multiply Value channel by random factor between 0.5 and 1.5
        random_brightness = np.random.uniform(0.5, 1.5)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random_brightness, 0, 255).astype(np.uint8)
        
        aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    return aug_img, aug_steering


def prepare_dataset(data_path, test_size=0.2, random_state=42):
    """
    Loads driving log and splits dataset into training and validation DataFrames.

    Parameters:
        data_path (str): Path to data directory or driving_log.csv.
        test_size (float): Proportion of dataset to include in validation split.
        random_state (int): Seed for reproducible train/test split.

    Returns:
        tuple: (train_dataframe, validation_dataframe)
    """
    df = load_driving_data(data_path)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    return train_df, val_df


class DrivingDataGenerator(tf.keras.utils.Sequence):
    """
    Efficient batch data generator subclassing tf.keras.utils.Sequence.
    """
    def __init__(self, dataframe, data_dir, batch_size=32, is_training=True):
        self.df = dataframe.reset_index(drop=True)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.is_training = is_training
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        steering_angles = []

        for i in batch_indices:
            row = self.df.iloc[i]
            center_path = row['center']
            steering = float(row['steering'])

            resolved_path = resolve_image_path(center_path, self.data_dir)
            img = cv2.imread(resolved_path)

            if img is None:
                continue

            # Apply full preprocessing (BGR->RGB, Crop, Resize, Normalize)
            img = preprocess_image(img)

            # Apply Data Augmentation for training dataset
            if self.is_training:
                img, steering = augment_image(img, steering)

            images.append(img)
            steering_angles.append(steering)

        if len(images) == 0:
            # Fallback for empty batch edge case
            dummy_img = np.zeros(INPUT_SHAPE, dtype=np.float32)
            return np.array([dummy_img]), np.array([0.0])

        return np.array(images, dtype=np.float32), np.array(steering_angles, dtype=np.float32)

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)
