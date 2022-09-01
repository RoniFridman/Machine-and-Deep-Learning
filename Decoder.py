import torch
from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, latent_dims, size):
        super(Decoder, self).__init__()
        self.img_size = size
        ndf = size
        self.ndf = ndf
        self.latent_dims = latent_dims
        self.fc = nn.Linear(latent_dims, ndf * 16)
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 16, ndf * 8, kernel_size=3, stride=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=3, stride=2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 4, ndf*2, kernel_size=3, stride=2),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=3, stride=2),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ndf, 3, kernel_size=4, stride=2),
            nn.Sigmoid())

    def forward(self, z):
        b_size = z.shape[0]
        z = self.fc(z).view(b_size, (self.ndf//4)*self.img_size, 1, 1)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.layer5(z)
        return z.reshape((-1, 3, self.img_size, self.img_size))
