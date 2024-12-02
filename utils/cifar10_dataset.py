import os
import random
import torch
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class MaskedCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, mask_transform=None, num_masks=10):
        """
        CIFAR-10 Dataset with synthetic masks.
        :param root: Path to store the CIFAR-10 dataset.
        :param train: If True, load the training split; otherwise, the test split.
        :param transform: Transformation for the image.
        :param mask_transform: Transformation for the mask.
        :param num_masks: Number of random masks to generate per image.
        """
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        self.mask_transform = mask_transform
        self.num_masks = num_masks

    def __len__(self):
        # Return the length of the CIFAR-10 dataset
        return len(self.dataset)

    def generate_random_mask(self, size):
        """
        Generate a random irregular mask.
        :param size: Tuple (width, height) for the mask.
        :return: PIL Image of the mask.
        """
        if isinstance(size, torch.Tensor):  # Handle the case where size is a tensor
            size = tuple(size[-2:])  # Get (height, width) from tensor shape

        mask = Image.new("L", size, 0)  # Black background
        draw = ImageDraw.Draw(mask)
        for _ in range(random.randint(5, 15)):
            x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
            x2, y2 = random.randint(0, size[0]), random.randint(0, size[1])
            width = random.randint(10, 50)
            draw.line([x1, y1, x2, y2], fill=255, width=width)
        return mask


    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Ignore labels; image is already a PIL.Image object

        if self.transform:
            image = self.transform(image)  # Convert image to tensor

        # Ensure the mask is created with the correct size format
        if isinstance(image, torch.Tensor):
            size = tuple(image.shape[-2:])  # If image is a tensor, use shape for size (height, width)
        else:
            size = image.size  # If image is a PIL image, use .size (width, height)

        # Generate random mask
        mask = self.generate_random_mask(size)

        # Convert the mask to tensor and make sure it's of type float
        mask = transforms.ToTensor()(mask)  # Convert the PIL image mask to a tensor
        mask = mask.float()  # Now you can safely apply .float() on the tensor

        # Apply mask to image: image * (1 - mask)
        corrupted_image = image * (1 - mask)

        return corrupted_image, mask, image  # Return corrupted, mask, and original image









# Example usage:
if __name__ == "__main__":
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load CIFAR-10 Dataset
    train_dataset = MaskedCIFAR10(
        root="datasets/cifar10/",
        train=True,
        transform=transform,
        mask_transform=mask_transform
    )

    # Data Loader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Test the data loader
    for corrupted, mask, original in train_loader:
        print("Corrupted Image Shape:", corrupted.shape)
        print("Mask Shape:", mask.shape)
        print("Original Image Shape:", original.shape)
        break
