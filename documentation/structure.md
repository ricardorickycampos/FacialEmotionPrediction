# Data
### Dataset — FER-2013
48x48 grayscale images, pre-cropped to the face region.

### Split Sizes
- Train: 22,968 images
- Validation: 5,741 images (20% split from train folder)
- Test: 7,178 images
- Total: 35,887 images

### Class Distribution (Training Set)
- angry: 3,196 samples
- disgust: 349 samples 
- fear: 3,278 samples
- happy: 5,772 samples 
- neutral: 3,972 samples
- sad: 3,864 samples
- surprise: 2,537 samples

### Class Imbalance Handling

To correct for uneaven class distribution, class weights are computed using sklearn `compute_class_weight('balanced')` and passed to the training loop. This penalizes mistakes on minority classes more heavily during backpropagation.

Computed class weights:

- angry: 1.027
- disgust: 9.402
- fear: 1.001
- happy: 0.568
- neutral: 0.826
- sad: 0.849
- surprise: 1.293


# data_loader.py

Handles all data loading, preprocessing, and exploration of data set

## Constants
- `IMG_SIZE = 48` — target image size (48x48 pixels)
- `BATCH_SIZE = 32` — number of images per training batch
- `NUM_CLASSES = 7` — angry, disgust, fear, happy, neutral, sad, surprise

## Functions

### `load_data()`
Loads and preprocesses the FER-2013 dataset from the folder structure.
- Normalizes pixel values from 0-255 to 0-1
- Splits training data 80/20 into train and validation sets
- Applies augmentation to training data only (horizontal flip, rotation, zoom)
- Returns three generators: `train_data`, `val_data`, `test_data`

### `get_class_weights(train_data)`
Computes class weights to handle class imbalance in the dataset.
- Underrepresented classes receive higher weights so the model penalizes mistakes on them more heavily during training
- Returns a dictionary mapping class index to weight value

### `explore_data(train_data)`
Prints class distribution and displays a 3x3 grid of sample training images.
- Shows class indices and sample counts per emotion
- Used for initial dataset exploration only, not part of training pipeline

# model.py

Defines the CNN architecture for facial emotion classification.

## Architecture Overview
A 3-block convolutional neural network and a fully connected 
classifier head. Built using Keras Sequential.

## Input
- Shape: `(48, 48, 1)` — grayscale 48x48 pixel image

## Output
- Shape: `(7,)` — probability distribution across 7 emotion classes
- Classes: angry, disgust, fear, happy, neutral, sad, surprise

## Layer Blocks

### Block 1 (32 filters)
- 2x Conv2D(32) with ReLU activation and same padding
- BatchNormalization after each Conv2D
- MaxPooling2D(2,2) — 
- Dropout(0.25) — 

### Block 2 (64 filters)
- 2x Conv2D(64) with ReLU activation and same padding
- BatchNormalization after each Conv2D
- MaxPooling2D(2,2)
- Dropout(0.25)

### Block 3 (128 filters)
- 2x Conv2D(128) with ReLU activation and same padding
- BatchNormalization after each Conv2D
- MaxPooling2D(2,2)
- Dropout(0.25)

### Classifier Head
- Flatten - 3D to 1D Vector
- Dense(256) with ReLU 
- BatchNormalization
- Dropout(0.5)
- Dense(7) with Softmax — outputs probability per emotion class

## Design Decisions
- `padding='same'` preserves spatial dimensions through Conv layers
- Filter progression 32→64→128 reflects increasing pattern complexity
- BatchNormalization after every Conv2D stabilizes training and acts as mild regularization

## Model Size
- Total parameters: ~1.2M
- Size on disk: ~5MB

# predict.py

Handles face detection and emotion prediction on a single static image.
This module is designed for testing and conneting static image inputs to the model.

## Dependencies
- OpenCV — image loading and drawing
- MediaPipe — face detection using BlazeFace model
- Keras — loading trained emotion model
- NumPy — array operations

## Constants
- `EMOTIONS` — ordered list of 7 emotion class labels matching model output indices
  `['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']`

## Functions

### `preprocess_input(img)`
Preprocesses a raw BGR image for model consumption.
1. Converts BGR to grayscale
2. Resizes to 48x48
3. Normalizes pixel values from 0-255 to 0-1
4. Reshapes to `(1, 48, 48, 1)` to match model input shape
- Returns: numpy array of shape `(1, 48, 48, 1)`

### `face_detect_predict(model, img)`
Detects the first face in an image and predicts its emotion.

**Face Detection:**
- Uses MediaPipe BlazeFace 
- Converts image from BGR to RGB for MediaPipe compatibility
- Applies 20px padding around detected bounding box for better context
- Returns early with `None` if no face is detected

**Emotion Prediction:**
- Crops detected face region from original image
- Passes crop through `preprocess_input()`
- Runs inference with `model.predict()`
- Takes argmax of 7 output probabilities to get predicted class
- Returns: `(emotion_label, confidence, bounding_box)` or `(None, None, None)`

## Usage
Loads model from `models/emotion_model.keras` and a test image,
draws bounding box and emotion label with confidence on the image,
and displays result in an OpenCV window.

# train.py

Handles model training, callback management, evaluation, and 
training history visualization.

## Dependencies
- Keras — model compilation, training, and callbacks
- Matplotlib — plotting training history
- data_loader — load_data(), get_class_weights()
- model — build_model()

## Functions

### `train_model()`
Full training pipeline from data loading to evaluation.

**Setup:**
- Loads train, val, and test generators from `data_loader.load_data()`
- Computes class weights from `data_loader.get_class_weights()` to 
  handle FER-2013 class imbalance
- Builds model from `model.build_model()`
- Compiles with Adam optimizer (lr=0.001), categorical crossentropy 
  loss, and accuracy metric

**Callbacks:**

| Callback | Monitors | Behavior |
|---|---|---|
| ModelCheckpoint | val_accuracy | Saves model only when val accuracy improves |
| EarlyStopping | val_accuracy | Stops training if no improvement for 8 epochs, restores best weights |
| ReduceLROnPlateau | val_loss | Halves learning rate if val loss plateaus for 5 epochs, min lr=1e-6 |

**Training:**
- Max 100 epochs with early stopping
- Class weights passed to compensate for imbalanced dataset
- Validation data monitored every epoch

**Evaluation:**
- Runs `model.evaluate()` on held out test set after training
- Prints final test accuracy and loss

- Returns: `(model, history)`

### `plot_history(history)`
Plots training and validation accuracy and loss curves side by side.
- Saves figure to `models/training_history.png`
- Used to visually diagnose overfitting and training behavior

## Training Results
- Best epoch: 56
- Test Accuracy: 63.25%
- Test Loss: 1.0014
