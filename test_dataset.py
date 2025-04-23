import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.dataset import CKPlusDataset

# Define transforms
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Initialize dataset
dataset = CKPlusDataset(root_dir="data/CK+48", transform=transform)

# Use a DataLoader to batch samples
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load a batch
data_iter = iter(dataloader)
image, label = next(data_iter)

# Show shape and label
print(f"Image shape: {image.shape}")
print(f"Label: {label.item()}")

# Display image
plt.imshow(image[0][0], cmap="gray")
plt.title(f"Label: {label.item()}")
plt.axis("off")
plt.show()
