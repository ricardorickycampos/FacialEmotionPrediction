import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Data directories
TRAIN_DIR = 'data/RAF/train'
TEST_DIR = 'data/RAF/test'

# Constants
IMG_SIZE = 100
BATCH_SIZE = 32 
NUM_CLASSES = 7 # 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral

def load_data():
    # Generate training/test data
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True, 
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8,1.2],
        )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
    )

    # Load data from data_gens
    train_data = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE), 
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training',
        seed=42,
    )

    val_data = val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        seed=42,
    )

    return train_data, val_data

# Balancing class weights due to diffrence in number of data values per class.
def get_class_weights(train_data): 
    labels = train_data.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(class_weights))

def explore_data(train_data):
    class_indices = train_data.class_indices
    print("Class indices:", class_indices)

    labels = train_data.classes
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(class_indices.keys(), counts):
        print(f"{cls}: {count} samples")

    # Display some sample images
    images, labels = next(train_data)
    fig, axes = plt.subplots(3,3, figsize=(8,8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        emotion = list(class_indices.keys())[np.argmax(labels[i])]
        ax.set_title(emotion)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_data, val_data = load_data()
    explore_data(train_data)

    class_weights = get_class_weights(train_data)
    print("\nClass weights: ")
    class_indices=train_data.class_indices
    for emotion, idx in class_indices.items():
        print(f"{emotion}: {class_weights[idx]:.3f}")