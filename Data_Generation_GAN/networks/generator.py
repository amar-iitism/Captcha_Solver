import torch.nn as nn


class Generator_Network(nn.Module):
    def __init__(self, img_size, latent_size, hidden_size):
        super(Generator_Network, self).__init__()

        self.latent_size = latent_size

        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, img_size)  # Output without activation function
        )

        # Calculate the output shape after the last linear layer
        self.output_shape = (img_size,)

    def forward(self, z):
        output = self.model(z)
        # Reshape the output to the desired image size
        output = output.view(-1, *self.output_shape)
        return output
