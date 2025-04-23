import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, confusion_matrix, precision_score, recall_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from utils.dataset import CKPlusDataset
from models.cnn import EmotionCNN

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

# Transforms
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset and model
dataset = CKPlusDataset(root_dir="data/CK+48", transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = EmotionCNN(num_classes=7)
model.load_state_dict(torch.load("emotion_cnn.pth", map_location=device))
model.to(device)
model.eval()

# Evaluation
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.append(labels.item())
        y_pred.append(preds.item())

# Accuracy report
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
for row in cm:
    print(row.tolist())  # To match his format

print("Precision:", precision_score(y_true, y_pred, average='macro'))
print("Recall:   ", recall_score(y_true, y_pred, average='macro'))
print("Accuracy: ", accuracy_score(y_true, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()