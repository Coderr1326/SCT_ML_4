# predict_gesture.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# match these with training
img_height, img_width = 64, 64

# Load trained model
model = tf.keras.models.load_model("hand_gesture_cnn.h5")

# Class names must match training indices
CLASS_NAMES = [
    "0", "1", "10", "11", "12", "13", "14", "15", "16", "17",
    "18", "19", "2", "3", "4", "5", "6", "7", "8", "9"
]
# or, better, print(train_gen.class_indices) during training and paste keys in that order

def predict_gesture(path):
    img = image.load_img(path, target_size=(img_height, img_width))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    idx = np.argmax(preds[0])
    label = CLASS_NAMES[idx]
    print(f"Predicted gesture: {label} (index {idx})")

if __name__ == "__main__":
    # change this to any test image
    test_image_path = "test_image.jpg"
    predict_gesture(test_image_path)
