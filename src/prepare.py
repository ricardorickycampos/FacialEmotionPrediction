import cv2 as cv
import mediapipe as mp
from pathlib import Path

TRAIN_INPUT_DIR = 'data/KDEF_split/train'
TEST_INPUT_DIR = 'data/KDEF_split/test'
TRAIN_CROPPED_DIR = 'data/KDEF_cropped/train'
TEST_CROPPED_DIR = 'data/KDEF_cropped/test'
CROPSIZE = 48

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='models/detector.tflite'),
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.4
)

def crop_face(detector, img, padding=4):
    rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_img)

    if not results.detections:
        return None

    h, w = img.shape[:2]
    bbox = results.detections[0].bounding_box

    x = max(0, int(bbox.origin_x) - padding)
    y = max(0, int(bbox.origin_y) - padding)
    bw = min(w - x, int(bbox.width) + 2 * padding)
    bh = min(h - y, int(bbox.height) + 2 * padding)

    crop = img[y:y+bh, x:x+bw]
    if crop.size == 0:
        return None

    return cv.resize(crop, (CROPSIZE, CROPSIZE), interpolation=cv.INTER_AREA)


def process_dataset(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    image_paths = list(src_dir.rglob('*.png')) + list(src_dir.rglob('*.jpg'))

    detected = 0
    fallback = 0
    failed = 0

    with FaceDetector.create_from_options(options) as detector:
        for i, path in enumerate(image_paths):
            if i % 500 == 0:
                print(f"{i}/{len(image_paths)} — detected: {detected}, fallback: {fallback}")

            img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
            if img is None:
                failed += 1
                continue

            relative = path.relative_to(src_dir)
            out_path = dst_dir / relative
            out_path.parent.mkdir(parents=True, exist_ok=True)

            cropped = crop_face(detector, img)

            if cropped is not None:
                cv.imwrite(str(out_path), cropped)
                detected += 1
            else:
                cv.imwrite(str(out_path), img)
                fallback += 1

    print(f"\nDone. Detected: {detected} | Fallback: {fallback} | Failed to load: {failed}")


if __name__ == "__main__":
    process_dataset(TRAIN_INPUT_DIR, TRAIN_CROPPED_DIR)
    process_dataset(TEST_INPUT_DIR, TEST_CROPPED_DIR)