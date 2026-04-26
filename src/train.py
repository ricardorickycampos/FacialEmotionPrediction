import os 
import matplotlib.pyplot as plt
from data_loader import load_data, get_class_weights
from model import build_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

MODEL_PNG = 'models/training_history_RAF.png'
MODEL_CHECKPOINT_PATH ='models/emotion_model_RAF.keras'

def train_model():
  train, val = load_data()
  class_weights = get_class_weights(train)

  emotion_model = build_model()
  emotion_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  callbacks = [
    ModelCheckpoint(
      filepath=MODEL_CHECKPOINT_PATH,
      monitor='val_accuracy',
      save_best_only=True,
      verbose=1
    ),
    EarlyStopping(
      monitor="val_accuracy",
      patience=9, 
      restore_best_weights=True,
      verbose=1
    ),
    ReduceLROnPlateau(
      monitor="val_loss",
      patience=5,
      min_lr=1e-6,
      verbose=1
    )
  ]

  history = emotion_model.fit(
    train,
    validation_data=val,
    epochs=100,
    class_weight=class_weights,
    callbacks=callbacks
  )

  print("\nEvaluation of test set: ")
  test_loss, test_accuracy = emotion_model.evaluate(val)
  print(f"Test Accuracy: {test_accuracy:.4f}")
  print(f"Test Loss: {test_loss:.4f}")

  return emotion_model, history

def plot_history(history):
  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14, 5))

  ax1.plot(history.history['accuracy'], label='Train Accuracy')
  ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
  ax1.set_title('Model Accuracy')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Accuracy')
  ax1.legend()

  ax2.plot(history.history['loss'], label='Train Loss')
  ax2.plot(history.history['val_loss'], label='Val Loss')
  ax2.set_title('Model Loss')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Loss')
  ax2.legend()

  plt.tight_layout()
  plt.savefig(MODEL_PNG)
  plt.show()
   

if __name__ == "__main__":
    model, history = train_model()
    plot_history(history)
