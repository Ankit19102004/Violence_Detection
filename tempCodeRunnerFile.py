import cv2
from keras.models import model_from_json
import numpy as np

# Load model
with open("cnn_model_224.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("cnn_model_224.h5")
print("✅ Model loaded successfully")

# Preprocessing
def extract_features(image):
    image = cv2.resize(image, (224, 224))
    image = np.array(image).reshape(1, 224, 224, 3) / 255.0
    return image

# Webcam
webcam = cv2.VideoCapture(0)
labels = {0: 'Accident', 1: 'Fight', 2: 'Fire', 3: 'Snatch', 4: 'Normal'}

frame_count = 0
predicted_label = "Normal"

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    frame_count += 1

    # Predict every 5th frame for speed
    if frame_count % 5 == 0:
        img = extract_features(frame)  # use whole frame, not just face
        pred = model.predict(img, verbose=0)
        predicted_label = labels[pred.argmax()]

    # Show prediction
    cv2.putText(frame, predicted_label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
