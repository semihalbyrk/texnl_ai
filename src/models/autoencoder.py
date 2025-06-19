import torch, torch.nn as nn

class BetaVAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim=8, beta=2.0):
        super().__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.mu     = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, in_dim)
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h  = self.encoder(x)
        mu, lv = self.mu(h), self.logvar(h)
        z  = self.reparam(mu, lv)
        return self.decoder(z), mu, lv

    def loss(self, x, recon, mu, lv):
        recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
        kld = -0.5*torch.mean(1 + lv - mu.pow(2) - lv.exp())
        return recon_loss + self.beta * kld
