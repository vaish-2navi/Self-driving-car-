import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from model import build_model
from utils import prepare_dataset, DrivingDataGenerator

# ==============================================================================
# Configurable Constants
# ==============================================================================
DATA_PATH = 'data'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = 'model.keras'
PLOT_SAVE_PATH = 'training_loss.png'


def main():
    print("=" * 65)
    print("  Behavioral Cloning Model Training - Udacity Self-Driving Car  ")
    print("=" * 65)
    print("Configuration Parameters:")
    print(f"  - DATA_PATH:        {DATA_PATH}")
    print(f"  - BATCH_SIZE:       {BATCH_SIZE}")
    print(f"  - EPOCHS:           {EPOCHS}")
    print(f"  - LEARNING_RATE:    {LEARNING_RATE}")
    print(f"  - MODEL_SAVE_PATH:  {MODEL_SAVE_PATH}")
    print("=" * 65)

    # 1. Load dataset & perform train/validation split
    print("\n[1/5] Loading driving dataset and splitting train/validation sets...")
    try:
        train_df, val_df = prepare_dataset(DATA_PATH, test_size=0.2, random_state=42)
        print("Dataset successfully loaded!")
        print(f"  - Total Center Camera Samples: {len(train_df) + len(val_df)}")
        print(f"  - Training Samples:            {len(train_df)}")
        print(f"  - Validation Samples:          {len(val_df)}")
    except FileNotFoundError as err:
        print(f"\n[ERROR] {err}")
        print("Please ensure driving_log.csv and IMG/ folder are present in the DATA_PATH directory.")
        return

    # 2. Initialize tf.keras Sequence Data Generators
    print("\n[2/5] Initializing tf.keras DataGenerators...")
    train_generator = DrivingDataGenerator(train_df, DATA_PATH, batch_size=BATCH_SIZE, is_training=True)
    val_generator = DrivingDataGenerator(val_df, DATA_PATH, batch_size=BATCH_SIZE, is_training=False)
    print("Data Generators ready.")

    # 3. Build & Compile NVIDIA CNN Model
    print("\n[3/5] Building NVIDIA End-to-End CNN Architecture...")
    model = build_model(input_shape=(66, 200, 3), learning_rate=LEARNING_RATE)
    model.summary()

    # 4. Set up callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    print("\n[4/5] Configuring Keras training callbacks...")
    checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    callbacks = [checkpoint, early_stopping, reduce_lr]

    # 5. Execute model training
    print("\n[5/5] Launching model training loop...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "=" * 65)
    print(f"Training successfully completed! Best model saved to: {MODEL_SAVE_PATH}")
    print("=" * 65)

    # Plot training & validation loss history
    print("\nGenerating training and validation loss plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss (MSE)', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', linewidth=2)
    plt.title('NVIDIA CNN Behavioral Cloning Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    print(f"Loss plot saved to: {PLOT_SAVE_PATH}")
    plt.show()


if __name__ == '__main__':
    main()
