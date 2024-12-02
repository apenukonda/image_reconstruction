import torch
import torch.optim as optim
from models.coarse_fine_generator import CoarseToFineGenerator
from models.cr_loss import CRLoss
from torch.utils.data import DataLoader
from utils.cifar10_dataset import MaskedCIFAR10
from torchvision import transforms

# Initialize dataset and dataloaders
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = MaskedCIFAR10(root="datasets/cifar10/", train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Check if CUDA is available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, and optimizer
model = CoarseToFineGenerator().to(device)  # Move model to the selected device
cr_loss = CRLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    for i, (corrupted, mask, original) in enumerate(train_loader):
        # Move data to the selected device
        corrupted, mask, original = corrupted.to(device), mask.to(device), original.to(device)

        # Forward pass
        generated = model(corrupted)

        # Calculate CR loss
        loss = cr_loss(generated, original, mask)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
