import torch
from torch import nn
import torch.nn.functional as F
from Encoder import Encoder
from Decoder import Decoder
from DiscreteVAE import DiscreteVAE
import numpy as np


class VariationalAutoencoder(nn.Module):
    def __init__(self,latent_dims, size):
        super().__init__()
        self.img_size = size
        self.encoder = Encoder(latent_dims, self.img_size)
        self.decoder = Decoder(latent_dims, self.img_size)
        self.discrete_vae = None

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
