import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size, hidden):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),
            nn.Sigmoid()  # Sigmoid activation for output
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
