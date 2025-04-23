import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import CKPlusDataset
from models.cnn import EmotionCNN

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs = 10
lr = 0.001

# Transforms
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset & Dataloader
dataset = CKPlusDataset(root_dir="data/CK+48", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
model = EmotionCNN(num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(dataloader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "emotion_cnn.pth")
print("✅ Model saved as emotion_cnn.pth")
