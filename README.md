# 😊 Facial Expression Recognition using CNN and CK+ Dataset

This project demonstrates a deep learning-based approach to classify facial expressions using a custom Convolutional Neural Network (CNN) trained on the CK+ dataset.

## 📌 Emotions Classified
- Anger
- Contempt
- Disgust
- Fear
- Happy
- Sadness
- Surprise

## 🧠 Approach
- **Input Size**: 48x48 grayscale images
- **Model**: Lightweight custom CNN
- **Tools**: TensorFlow, Keras, Python

## 📈 Results
| Metric        | Score     |
|---------------|-----------|
| Accuracy      | 96.95%    |
| Precision     | 94.47%    |
| Recall        | 97.36%    |
| F1-Score      | 96.07%    |


## 🚀 How to Run

```bash
git clone https://github.com/yourusername/ckplus-emotion-recognition.git
cd ckplus-emotion-recognition
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
