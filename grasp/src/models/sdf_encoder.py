import torch
import torch.nn as nn
from einops import rearrange


class VoxelSDFEncoder(nn.Module):
    def __init__(self):
        super(VoxelSDFEncoder, self).__init__()

        # Define the convolutional encoder
        self.encoder = nn.Sequential(
            # First 3D Conv Block
            # Input: (batch, 1, 48, 48, 48)  # Changed to accept 1 input channel
            # After Conv3d: (batch, 32, 48, 48, 48)
            nn.Conv3d(
                in_channels=1, out_channels=32, kernel_size=3, padding=1
            ),  # Changed in_channels to 1
            nn.ReLU(),
            # After MaxPool3d: (batch, 32, 24, 24, 24)
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Second 3D Conv Block
            # After Conv3d: (batch, 64, 24, 24, 24)
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # After MaxPool3d: (batch, 64, 12, 12, 12)
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Third 3D Conv Block
            # After Conv3d: (batch, 128, 12, 12, 12)
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # After MaxPool3d: (batch, 128, 6, 6, 6)
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Fourth 3D Conv Block
            # After Conv3d: (batch, 256, 6, 6, 6)
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            # After MaxPool3d: (batch, 256, 3, 3, 3)
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Global Average Pooling to reduce spatial dimensions
        # After AvgPool3d: (batch, 256, 1, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        # Add batch dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Add channel dimension

        # Validate input shape
        # assert x.shape[-4:] == torch.Size([1, 48, 48, 48]), (
        #     f"Expected shape (..., 1, 48, 48, 48), got {x.shape}"
        # )
        x = x.cuda()
        # print('---- x ----,', x.device)

        # Input shape: (batch_size, 1, 48, 48, 48)
        x = self.encoder(x)
        # Shape after encoder: (batch_size, 256, 3, 3, 3)

        x = self.avg_pool(x)
        # Shape after avg_pool: (batch_size, 256, 1, 1, 1)

        # Flatten using einops: (batch, channels, 1, 1, 1) -> (batch, channels)
        x = rearrange(x, "b c d h w -> b (c d h w)")
        # Shape after flatten: (batch_size, 256)
        return x


if __name__ == "__main__":
    # Create a sample input tensor
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 48, 48, 48)  # Updated test input shape

    # Initialize the model
    model = VoxelSDFEncoder()

    # Forward pass
    output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # Will be (batch_size, 256)
