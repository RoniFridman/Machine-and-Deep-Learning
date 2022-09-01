import torch
from torch import nn
import torch.nn.functional as F
from GumbleTrick import gumbel_softmax


class DiscreteVAE(nn.Module):
    def __init__(self, latent_dim, categorical_dim, size):
        super(DiscreteVAE, self).__init__()
        ndf = categorical_dim * 2
        self.N = latent_dim//2
        self.K = categorical_dim
        self.img_size = size
        self.b_size = None
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=4, stride=2),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*2 * self.N, kernel_size=4, stride=2),
            nn.ReLU())

        self.layer31 = nn.Sequential(
            nn.Conv2d(ndf*2 * self.N, ndf*4 * self.N, kernel_size=4, stride=2),
            nn.ReLU())

        self.fc1 = nn.Linear(ndf*4*4 * self.N, categorical_dim * self.N)
        self.fc2 = nn.Linear(categorical_dim * self.N, ndf*4*4 * self.N)

        ## Decoder
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ndf*4 * self.N, ndf*2 * self.N, kernel_size=4, stride=2),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ndf*2 * self.N, ndf * 2, kernel_size=4, stride=2),
            nn.ReLU())

        self.layer51 = nn.Sequential(
            nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=4, stride=2),
            nn.ReLU())

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(ndf, 3, kernel_size=6, stride=2),
            nn.Sigmoid())


    def encoder(self, x):
        b_size = x.shape[0]
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h3 = self.layer31(h3)
        x = h3.view(b_size, -1)
        x = self.fc1(x)
        return x

    def decoder(self, z):
        b_size = z.shape[0]
        z = z.view(b_size, -1)
        z = self.fc2(z)
        z = z.view(b_size, self.K*8 * self.N, 2,2)
        h4 = self.layer4(z)
        h5 = self.layer5(h4)
        h5 = self.layer51(h5)
        h6 = self.layer6(h5)
        return h6

    def loss_function(self, recon_x, x, qy):
        BCE = F.binary_cross_entropy(recon_x, x.view(self.b_size, 3, self.img_size, self.img_size), reduction='sum') / \
              x.shape[0]

        log_ratio = torch.log(qy * qy.size(-1) + 1e-20)
        KLD = torch.sum(qy * log_ratio, dim=-1).mean()

        return BCE, KLD

    def forward(self, x, temp, hard):
        self.b_size = x.shape[0]
        q = self.encoder(x)
        q_y = q.view(self.b_size, self.N, self.K)
        z = gumbel_softmax(q_y, temp, hard)
        return self.decoder(z), F.softmax(q_y, dim=-1).reshape(self.b_size, self.N, self.K)
