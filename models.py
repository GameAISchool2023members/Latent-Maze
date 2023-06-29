import torch
import torch.nn as nn
import torch.optim as optim

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

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        
        self.reconstruction_loss = nn.MSELoss()
        self.kl_divergence_loss = lambda mu, log_var: -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        learning_rate = 0.001
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        z_mean_log_var = self.encoder(x)
        z_mean = z_mean_log_var[:, :self.latent_size]
        z_log_var = z_mean_log_var[:, self.latent_size:]
        
        z = self.reparameterize(z_mean, z_log_var)
        
        reconstructed_x = self.decoder(z)
        
        return reconstructed_x, z_mean, z_log_var
    
    def loss_function(self, reconstructed_x, x, mu, log_var):
        recon_loss = self.reconstruction_loss(reconstructed_x, x)
        kld_loss = self.kl_divergence_loss(mu, log_var)
        return recon_loss + kld_loss
    
    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            for batch_data in dataloader:
                batch_data = batch_data[0]
                self.optimizer.zero_grad()
                
                reconstructed_x, z_mean, z_log_var = self(batch_data)
                
                loss = self.loss_function(reconstructed_x, batch_data, z_mean, z_log_var)
                
                loss.backward()
                self.optimizer.step()
                
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
