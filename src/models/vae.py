import numpy as np 
import torch 
from torch import nn 

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 10, img_size: torch.Size = torch.Size([1, 64, 64])):
        super().__init__()
        self.encoder = Encoder(latent_dim, img_size)
        self.decoder = Decoder(latent_dim, img_size)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    @property
    def kl(self):
        return self.encoder.kl

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 10, img_size: torch.Size = torch.Size([1, 64, 64])):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.channels = img_size[0]
        
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 4 * 4, 256)
        )
        
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return z, mu, logvar
        

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 10, img_size: torch.Size = torch.Size([1, 64, 64])):
        super().__init__()
        self.channels = img_size[0]

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)