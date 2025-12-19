# SCT_ML_4
# Hand Gesture Recognition (Digits 0–19) 🖐️

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.1-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a **Convolutional Neural Network (CNN)** to recognize hand gesture images representing digits from **0 to 19**. The trained model achieves near-perfect accuracy on the test set and can be used for gesture-based control or human-computer interaction.

## ✨ Features

- ✅ 20-class hand gesture recognition (digits 0-19)
- ✅ Data augmentation for robust training
- ✅ 80/20 train-validation split
- ✅ CNN architecture with dropout regularization
- ✅ Training curves visualization
- ✅ Saved model for inference (`hand_gesture_cnn.h5`)

## 📊 Dataset

**Hand Gesture Recognition Dataset** from Kaggle:
- **~18,000 training images** across 20 classes
- **Color RGB images** (64x64 pixels)
- Organized in `train/` and `test/` folders

### Download Dataset
https://www.kaggle.com/datasets/aryarishabh/hand-gesture-recognition-dataset



**Directory Structure:**
dataset/
├── train/
│ ├── 0/ # ~900 images
│ ├── 1/
│ ├── ...
│ └── 19/
└── test/
├── 0/
├── 1/
├── ...
└── 19/


## 🏗️ CNN Architecture

Input (64x64x3)
↓
Conv2D(32, 3x3) → MaxPooling2D(2x2)
↓
Conv2D(64, 3x3) → MaxPooling2D(2x2)
↓
Conv2D(128, 3x3) → MaxPooling2D(2x2)
↓
Flatten
↓
Dense(256, ReLU) → Dropout(0.5)
↓
Dense(20, Softmax)


## 🚀 Quick Start

### 1. Clone Repository
git clone <your-repo-url>
cd hand-gesture-recognition


### 2. Install Dependencies
pip install tensorflow==2.13.1 matplotlib


### 3. Download & Setup Dataset
1. Download from [Kaggle Dataset](https://www.kaggle.com/datasets/aryarishabh/hand-gesture-recognition-dataset)
2. Extract to `dataset/` folder (or update paths in `hand_gesture_cnn.py`)

### 4. Train Model
python hand_gesture_cnn.py


## 📈 Expected Results

Training Images: ~14,400 (80% split)
Validation Images: ~3,600 (20% split)
Test Images: ~18,000

Test Accuracy: ~99-100%


**Generated Files:**
- `hand_gesture_cnn.h5` - Trained model (complete)
- `gesture_training_curves.png` - Training visualization

## 🛠️ Training Configuration

| Parameter | Value |
|-----------|-------|
| **Loss** | categorical_crossentropy |
| **Optimizer** | Adam |
| **Metrics** | accuracy |
| **Epochs** | 25 |
| **Batch Size** | 32 |
| **Image Size** | 64x64x3 |
| **Augmentation** | rotation, shift, zoom, flip |

## 🔮 Future Enhancements

- [ ] Real-time webcam inference
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Transfer learning with pre-trained models
- [ ] Custom gesture training pipeline

## 📝 Usage Example

from tensorflow.keras.models import load_model
model = load_model('hand_gesture_cnn.h5')

Predict single image
prediction = model.predict(img_array)
gesture_class = np.argmax(prediction)
print(f"Predicted digit: {gesture_class}")


## 🐛 Troubleshooting

1. **CUDA/GPU Issues**: Install CPU version `tensorflow-cpu==2.13.1`
2. **Memory Issues**: Reduce `batch_size` in script
3. **Dataset Path**: Update `data_dir` in `hand_gesture_cnn.py`

## 📄 Citation

@misc{hand_gesture_dataset,
author = {Arya Rishabh},
title = {Hand Gesture Recognition Dataset},
year = {2021},
publisher = {Kaggle},
url = {https://www.kaggle.com/datasets/aryarishabh/hand-gesture-recognition-dataset}
}


## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hand Gesture Recognition Dataset](https://www.kaggle.com/datasets/aryarishabh/hand-gesture-recognition-dataset) by Arya Rishabh
- TensorFlow/Keras documentation and examples

---

⭐ **Star this repository if you found it helpful!** ⭐
