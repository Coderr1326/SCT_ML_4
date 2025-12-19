# hand_gesture_cnn.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


img_height, img_width = 64, 64   
batch_size = 32
num_classes = 20                 
train_dir = "dataset/train"
test_dir = "dataset/test"       


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,        
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

print("Class indices:", train_gen.class_indices)

test_gen = None
if os.path.exists(test_dir):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu",
                  input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

epochs = 25

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen
)

if test_gen is not None:
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc:.4f}")

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Train Acc")
plt.plot(epochs_range, val_acc, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.tight_layout()
plt.savefig("gesture_training_curves.png", dpi=300)
plt.show()

print("Training curves saved as 'gesture_training_curves.png'.")

model.save("hand_gesture_cnn.h5")
print("Model saved as 'hand_gesture_cnn.h5'.")

CLASS_NAMES = list(train_gen.class_indices.keys())

def predict_gesture(image_path):
    from tensorflow.keras.preprocessing import image

    img = image.load_img(image_path, target_size=(img_height, img_width))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    idx = np.argmax(preds[0])
    gesture_label = CLASS_NAMES[idx]
    print(f"Predicted gesture: {gesture_label}  (index {idx})")
    return gesture_label

