import os
import tensorflow as tf
from mtcnn import MTCNN
import cv2
from PIL import Image

# Check for GPU availability
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Input directories
REAL_DIR = r"celebdb-dataset/celebdb Real Images"
FAKE_DIR = r"celebdb-dataset/celebdb-Fake-Images Copy"
# Output directories
REAL_OUT = r"celebdb-dataset/real_faces"
FAKE_OUT = r"celebdb-dataset/fake_faces"

os.makedirs(REAL_OUT, exist_ok=True)
os.makedirs(FAKE_OUT, exist_ok=True)

def detect_and_crop_faces(input_dir, output_dir):
    detector = MTCNN()
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        try:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(img)
            if results:
                # Take the largest face detected
                largest = max(results, key=lambda x: x['box'][2] * x['box'][3])
                x, y, w, h = largest['box']
                x, y = max(0, x), max(0, y)
                cropped_face = img[y:y+h, x:x+w]
                face_img = Image.fromarray(cropped_face)
                face_img.save(os.path.join(output_dir, img_name))
            else:
                print(f"No face detected in {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    print("Processing real images...")
    detect_and_crop_faces(REAL_DIR, REAL_OUT)
    print("Processing fake images...")
    detect_and_crop_faces(FAKE_DIR, FAKE_OUT)
    print("Done. Cropped faces are saved in 'real_faces' and 'fake_faces'.")
