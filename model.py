import torch.nn as nn
import torch.nn.functional as F
import torch


class LinearModelGenerator(nn.Module):
    def __init__(self, latent_dim):
        super(LinearModelGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 4 * 32 * 32),
            nn.BatchNorm1d(4 * 32 * 32),
            nn.ReLU(),
            nn.Linear(4 * 32 * 32, 2 * 32 * 32),
            nn.BatchNorm1d(2 * 32 * 32),
            nn.ReLU(),
            nn.Linear(2 * 32 * 32, 32 * 32),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)

        return x.view(-1, 1, 32, 32)


class DCGANModelGenerator(nn.Module):
    def __init__(self, latent_dim):
        super(DCGANModelGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.project = nn.Linear(latent_dim, 4 * 4 * 256)
        # self.bn = nn.BatchNorm1d(4 * 4 * 256)

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.project(x)
        x = F.relu(x)
        x = self.conv(x.view(-1, 256, 4, 4))

        return x


class DCGANModelDiscriminator(nn.Module):
    def __init__(self):
        super(DCGANModelDiscriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=2),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.Dropout2d(0.5),
            nn.Conv2d(64, 128, 3, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.Dropout2d(0.5),
            nn.Conv2d(128, 256, 3, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.Dropout2d(0.5),
            nn.Conv2d(256, 1, 4, stride=4),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)

        return x.squeeze()
