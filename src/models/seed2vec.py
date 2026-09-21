import torch
import torch.nn as nn
import torch.nn.functional as F


class Seed2Vec(nn.Module):
    """VAE mapping binary seed vector to/from latent space z."""

    def __init__(self, num_nodes, latent_dim=64, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def decode(self, z):
        return torch.sigmoid(self.dec(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar