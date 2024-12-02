import torch
import torch.nn as nn
import torch.nn.functional as F

class CoarseToFineGenerator(nn.Module):
    def __init__(self):
        super(CoarseToFineGenerator, self).__init__()

        # Coarse network
        self.coarse_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Reduced to 32 channels
        self.coarse_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Reduced to 64 channels

        # Adjust the input size of the first fully connected layer based on the flattened output size
        self.coarse_fc1 = nn.Linear(128 * 64 * 64, 512)  # Adjusted to match flattened size
        self.coarse_fc2 = nn.Linear(512, 3 * 64 * 64)  # Output image size (3, 64, 64)

        # Refinement network
        self.refine_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.refine_conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.refine_fc1 = nn.Linear(128 * 64 * 64, 512)  # Adjusted to match flattened size
        self.refine_fc2 = nn.Linear(512, 3 * 64 * 64)  # Output image size (3, 64, 64)

    def forward(self, x):
        # Coarse network forward pass
        x_coarse = F.relu(self.coarse_conv1(x))
        x_coarse = F.relu(self.coarse_conv2(x_coarse))

        # Flatten the output of the convolutional layers
        x_coarse = x_coarse.view(x_coarse.size(0), -1)  # Flatten: (batch_size, num_features)

        x_coarse = F.relu(self.coarse_fc1(x_coarse))
        coarse_output = self.coarse_fc2(x_coarse).view(x.size(0), 3, 64, 64)

        # Refinement network forward pass
        x_refine = F.relu(self.refine_conv1(coarse_output))
        x_refine = F.relu(self.refine_conv2(x_refine))

        # Flatten the output of the refinement network
        x_refine = x_refine.view(x_refine.size(0), -1)  # Flatten: (batch_size, num_features)

        x_refine = F.relu(self.refine_fc1(x_refine))
        refined_output = self.refine_fc2(x_refine).view(x.size(0), 3, 64, 64)

        return refined_output
