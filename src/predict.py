import cv2 as cv
import numpy as np
from keras.models import load_model
from collections import deque
import mediapipe as mp
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'emotion_model_cropped.keras')
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
#EMOTIONS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
SMOOTHING_FRAMES = 20
PREPROCESS_SIZE = 48

def preprocess_input(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (PREPROCESS_SIZE, PREPROCESS_SIZE), interpolation=cv.INTER_AREA)
    norm = img.astype('float') / 255
    return norm.reshape(1, PREPROCESS_SIZE, PREPROCESS_SIZE, 1)


def get_face_bbox(detector, img):
    h, w = img.shape[:2]
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    )
    results = detector.detect(mp_img)
    if not results.detections:
        return None

    bbox = results.detections[0].bounding_box
    padding = 90
    x = max(0, int(bbox.origin_x) - padding)
    y = max(0, int(bbox.origin_y) - padding)
    width = min(w - x, int(bbox.width) + 2 * padding)
    height = min(h - y, int(bbox.height) + 2 * padding)

    return (x, y, width, height)

def draw_prediction(frame, bbox, emotion, confidence):
    x, y, w, h = bbox
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label = f'{emotion} {confidence:.0%}'
    label_y = y + h + 25

    # Background strip for readability
    (text_w, text_h), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv.rectangle(frame, (x, y + h), (x + text_w + 6, y + h + 30), (0, 255, 0), -1)
    cv.putText(frame, label, (x + 3, label_y),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

def run_prediction_stream():
    model = load_model(MODEL_PATH)

    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(BASE_DIR, '..', 'models', 'detector.tflite')),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=0.5
    )
    prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    print("Press 'q' to quit.")

    with FaceDetector.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            bbox = get_face_bbox(detector, frame)

            if bbox:
                x, y, w, h = bbox
                face_crop = frame[y:y + h, x:x + w]

                try:
                    preprocessed = preprocess_input(face_crop)
                    # visualize what gets fed to the model
                    debug_frame = (preprocessed.reshape(PREPROCESS_SIZE, PREPROCESS_SIZE) * 255).astype(np.uint8)
                    cv.imshow('Model Input', debug_frame)

                    preprocessed = preprocess_input(face_crop)
                    predictions = model.predict(preprocessed, verbose=0)
                    prediction_buffer.append(predictions[0])
                except Exception:
                    pass  # skip bad crops (too small, etc.)

            if prediction_buffer and bbox:
                avg_predictions = np.mean(prediction_buffer, axis=0)
                emotion_index = np.argmax(avg_predictions)
                emotion = EMOTIONS[emotion_index]
                confidence = avg_predictions[emotion_index]
                draw_prediction(frame, bbox, emotion, confidence)

            cv.imshow('Emotion Detection', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    run_prediction_stream()