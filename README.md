# MULTICLASS-FACIAL-EMOTION-RECOGNITION-USING-ATTENTION-MECHANSIM-WITH-REGULARISED-STEP-DECAY-LEARNING# Multiclass Facial Emotion Recognition using Attention Mechanism, Regularizers, and Step Decay Learning

This project implements a deep learning model for **Facial Emotion Recognition (FER)** using an enhanced **Xception-based architecture**. The system incorporates **spatial attention mechanisms**, **L2 regularization**, and a **step decay learning rate schedule** to achieve high accuracy on multiclass emotion classification tasks.

## 🚀 Key Features

- ✅ Xception-based Convolutional Neural Network (CNN)
- ✅ Spatial Attention Layer for focusing on key facial regions
- ✅ L2 Regularization to prevent overfitting
- ✅ Step Decay Learning Rate for optimized training
- ✅ Data Augmentation using Keras `ImageDataGenerator`
- ✅ Trained on **CK+** and **FER2013** datasets
- ✅ Handles 7 emotion classes: `Anger`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`

## 🧠 Model Architecture

- **Entry Flow, Middle Flow, Exit Flow** (Xception-style)
- **Depthwise Separable Convolutions**
- **Spatial Attention Block**
- **Global Average Pooling + Dense Layer with Softmax**
- **Dropout and L2 Regularization**

## 📊 Training Strategy

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Learning Rate Schedule:** Step Decay
  - Phase 1: `0.001`  
  - Phase 2: `0.0001`  
  - Phase 3: `0.00001`
- **Regularization:** L2 penalty on final dense layer
- **Batch Size:** 64
- **Epochs:** 300 (3 blocks of 100 each)

## 🧪 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Loss/Accuracy Curves

## 📂 Dataset Used

1. **[CK+ Dataset](https://www.jeffcohn.net/Resources/)**  
   - Clean, high-quality expressions in controlled settings.

2. **[FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)**  
   - In-the-wild images with real-world noise and variation.
