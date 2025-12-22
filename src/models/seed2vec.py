# TODO: Implement logic here
import torch
import torch.nn as nn
import torch.nn.functional as F

class Seed2Vec(nn.Module):
    def __init__(self, num_nodes, latent_dim=64, hidden_dim=128, dropout=0.2):
        super(Seed2Vec, self).__init__()
        
        # --- Encoder ---
        # Input: Binary Vector x (kích thước = num_nodes)
        self.enc1 = nn.Linear(num_nodes, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output: Mean và Log-Variance của latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # --- Decoder ---
        # Input: Latent vector z
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, num_nodes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.enc1(x))
        h = self.dropout(h)
        h = self.relu(self.enc2(h))
        # Trả về mu và logvar
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization Trick: z = mu + sigma * epsilon"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        h = self.relu(self.dec1(z))
        h = self.dropout(h)
        h = self.relu(self.dec2(h))
        # Output dùng Sigmoid để đưa về khoảng [0, 1] (xác suất chọn node)
        return torch.sigmoid(self.dec_out(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar