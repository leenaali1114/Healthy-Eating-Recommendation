import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Set paths
DATASET_PATH = 'dataset'  # Main dataset folder
MODEL_PATH = 'model/model.h5'

# Image parameters
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32

def create_model():
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        MaxPooling2D(2, 2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Flatten and dense layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (fresh or rotten)
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return None, None
    
    # Print dataset structure
    print("Dataset structure:")
    for root, dirs, files in os.walk(DATASET_PATH):
        level = root.replace(DATASET_PATH, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:  # Don't print all files, just the structure
            for d in dirs:
                print(f"{indent}    {d}/")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Only rescaling for validation
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    print("Loading training data...")
    
    # Try to automatically detect the dataset structure
    train_dir = DATASET_PATH
    
    # Check if there's a 'train' subdirectory
    if os.path.exists(os.path.join(DATASET_PATH, 'train')):
        train_dir = os.path.join(DATASET_PATH, 'train')
        print(f"Found training directory: {train_dir}")
    
    # Load training data
    try:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training'
        )
        
        # Load validation data
        validation_generator = valid_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation'
        )
        
        print(f"Found {train_generator.samples} training images")
        print(f"Found {validation_generator.samples} validation images")
        print(f"Classes: {train_generator.class_indices}")
        
        # Create and train the model
        model = create_model()
        
        # Print model summary
        model.summary()
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        print("Training model...")
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // BATCH_SIZE,
            epochs=20,
            callbacks=[checkpoint, early_stopping]
        )
        
        # Save the model
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('model/training_history.png')
        plt.close()
        
        print("Training complete! History plot saved to model/training_history.png")
        
        return model, history
        
    except Exception as e:
        print(f"Error during training: {e}")
        
        # Try alternative dataset structure
        print("Trying alternative dataset structure...")
        
        # Check if the dataset has a different structure (e.g., fresh/rotten at the top level)
        if any(d in ['fresh', 'rotten', 'Fresh', 'Rotten'] for d in os.listdir(DATASET_PATH)):
            print("Found fresh/rotten folders at the top level")
            
            # Create a temporary directory structure
            import tempfile
            import shutil
            
            temp_dir = tempfile.mkdtemp()
            print(f"Created temporary directory: {temp_dir}")
            
            # Create train directory
            train_temp = os.path.join(temp_dir, 'train')
            os.makedirs(train_temp, exist_ok=True)
            
            # Create fresh and rotten subdirectories
            fresh_dir = os.path.join(train_temp, 'fresh')
            rotten_dir = os.path.join(train_temp, 'rotten')
            os.makedirs(fresh_dir, exist_ok=True)
            os.makedirs(rotten_dir, exist_ok=True)
            
            # Find and copy fresh images
            for d in os.listdir(DATASET_PATH):
                if d.lower() == 'fresh':
                    src_dir = os.path.join(DATASET_PATH, d)
                    for item in os.listdir(src_dir):
                        s = os.path.join(src_dir, item)
                        if os.path.isdir(s):
                            # If there are subdirectories, copy all images from them
                            for img in os.listdir(s):
                                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    shutil.copy(os.path.join(s, img), fresh_dir)
                        elif item.lower().endswith(('.png', '.jpg', '.jpeg')):
                            # Copy direct images
                            shutil.copy(s, fresh_dir)
            
            # Find and copy rotten images
            for d in os.listdir(DATASET_PATH):
                if d.lower() == 'rotten':
                    src_dir = os.path.join(DATASET_PATH, d)
                    for item in os.listdir(src_dir):
                        s = os.path.join(src_dir, item)
                        if os.path.isdir(s):
                            # If there are subdirectories, copy all images from them
                            for img in os.listdir(s):
                                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    shutil.copy(os.path.join(s, img), rotten_dir)
                        elif item.lower().endswith(('.png', '.jpg', '.jpeg')):
                            # Copy direct images
                            shutil.copy(s, rotten_dir)
            
            print(f"Fresh images: {len(os.listdir(fresh_dir))}")
            print(f"Rotten images: {len(os.listdir(rotten_dir))}")
            
            # Try training with the new structure
            try:
                train_generator = train_datagen.flow_from_directory(
                    train_temp,
                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                    batch_size=BATCH_SIZE,
                    class_mode='binary',
                    subset='training'
                )
                
                validation_generator = valid_datagen.flow_from_directory(
                    train_temp,
                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                    batch_size=BATCH_SIZE,
                    class_mode='binary',
                    subset='validation'
                )
                
                print(f"Found {train_generator.samples} training images")
                print(f"Found {validation_generator.samples} validation images")
                print(f"Classes: {train_generator.class_indices}")
                
                # Create and train the model
                model = create_model()
                
                # Print model summary
                model.summary()
                
                # Callbacks
                checkpoint = ModelCheckpoint(
                    MODEL_PATH,
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )
                
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                # Train the model
                print("Training model...")
                history = model.fit(
                    train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // BATCH_SIZE,
                    epochs=20,
                    callbacks=[checkpoint, early_stopping]
                )
                
                # Save the model
                model.save(MODEL_PATH)
                print(f"Model saved to {MODEL_PATH}")
                
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
                
                return model, history
                
            except Exception as nested_e:
                print(f"Error with alternative structure: {nested_e}")
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
        
        return None, None

if __name__ == "__main__":
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    print(f"Looking for dataset in: {DATASET_PATH}")
    model, history = train_model()
    
    if model is None:
        print("Training failed. Please check the dataset structure.")
        print("Expected structure:")
        print("dataset/")
        print("├── train/")
        print("│   ├── fresh/")
        print("│   │   ├── image1.jpg")
        print("│   │   ├── image2.jpg")
        print("│   │   └── ...")
        print("│   └── rotten/")
        print("│       ├── image1.jpg")
        print("│       ├── image2.jpg")
        print("│       └── ...")
        print("OR")
        print("dataset/")
        print("├── fresh/")
        print("│   ├── image1.jpg")
        print("│   ├── image2.jpg")
        print("│   └── ...")
        print("└── rotten/")
        print("    ├── image1.jpg")
        print("    ├── image2.jpg")
        print("    └── ...")
