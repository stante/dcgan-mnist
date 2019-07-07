import torch.nn as nn


class DCGANModelGenerator(nn.Module):
    def __init__(self, latent_dim):
        super(DCGANModelGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.project = nn.Linear(latent_dim, 4 * 4 * 1024)
        self.bn = nn.BatchNorm1d(4 * 4 * 1024)

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.bn(self.project(x))
        x = nn.ReLU(x)
        x = self.conv(self.conv1(x.view(4, 4, -1)))

        return x


class DCGANModelDiscriminator(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
