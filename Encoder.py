import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dims, size):
        super(Encoder, self).__init__()
        self.img_size = size
        self.latent_dims = latent_dims
        ndf = size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=3, stride=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf * 4, kernel_size=3, stride=2),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 16, kernel_size=4, stride=2),
            nn.ReLU())
        self.fc = nn.Linear(ndf * 16, ndf * 32)
        self.to_mean = nn.Linear(ndf * 32, latent_dims)
        self.to_var = nn.Linear(ndf * 32, latent_dims)

        self.kl = None

    @staticmethod
    def reparametrization_trick(mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn_like(log_var)
        std = torch.exp(log_var / 2)
        return mu + std * eps

    def forward(self, x):
        b_size = x.shape[0]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(b_size, -1)
        x = self.fc(x)
        mu = self.to_mean(x)
        log_var = self.to_var(x)
        self.kl = torch.sum(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        return Encoder.reparametrization_trick(mu, log_var)
