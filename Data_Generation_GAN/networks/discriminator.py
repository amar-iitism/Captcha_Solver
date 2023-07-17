import torch.nn as nn


class Discriminator_Network(nn.Module):
    def __init__(self, img_size, hidden_size):
        super(Discriminator_Network, self).__init__()

        self.img_size = img_size

        self.model = nn.Sequential(
            nn.Linear(img_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, 1)  # Output without activation function
        )

    def forward(self, x):
        # Resize x from a H x W img to a vector
        x = x.view(-1, self.img_size)
        return self.model(x).clamp(1e-9, 1 - 1e-9)