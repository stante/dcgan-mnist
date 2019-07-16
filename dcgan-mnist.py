import click
import torchvision.datasets.mnist as mnist
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model
import pickle
import numpy as np


@click.command()
@click.option('--root', default="~/.torch/mnist", help="Root directory for MNIST dataset")
@click.option('--epochs', default=100, help="Number of epochs")
@click.option('--batch-size', default=128, help="Batch size")
@click.option('--latent-vector', default=100, help="Size of latent vector Z")
@click.option('--disable-cuda', default=False, help="Disable CUDA acceleration")
def main(root, epochs, batch_size, latent_vector, disable_cuda):
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    transform = transforms.Compose([transforms.Resize(32),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x * 2 - 1)])
    trainset = mnist.MNIST(root=root, download=True, train=True, transform=transform)
    testset = mnist.MNIST(root=root, download=True, train=False, transform=transform)
    trainloader = dataloader.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = dataloader.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, drop_last=True)

    G = model.LinearModelGenerator(latent_vector)
    D = model.DCGANModelDiscriminator()
    G.to(device)
    D.to(device)

    g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()

    z_fixed = torch.rand((16, latent_vector), device=device) * 2 - 1
    images = []

    for epoch in range(1, epochs + 1):
        generator_loss = 0
        discriminator_loss = 0
        g_acc = 0
        skip = True
        for real_images, y_label in trainloader:
            real_images = real_images.to(device)

            # Discriminator
            if skip:
                d_optimizer.zero_grad()

                # z = torch.rand((batch_size, latent_vector), device=device) * 2 - 1
                z = np.random.uniform(-1, 1, size=(batch_size, latent_vector))
                z = torch.from_numpy(z).float().to(device)
                fake_images = G.forward(z)

                # Real images
                real_out = D.forward(real_images)
                real_d_loss = criterion(real_out, torch.ones_like(y_label, dtype=torch.float32, device=device) * 0.9)
                fake_out = D.forward(fake_images)
                fake_d_loss = criterion(fake_out, torch.zeros_like(y_label, dtype=torch.float32, device=device))
                d_loss = real_d_loss + fake_d_loss
                d_loss.backward()
                d_optimizer.step()

            # Generator
            g_optimizer.zero_grad()
            # z = torch.rand((batch_size, latent_vector), device=device) * 2 - 1
            z = np.random.uniform(-1, 1, size=(batch_size, latent_vector))
            z = torch.from_numpy(z).float().to(device)
            fake_images = G.forward(z)
            out = D.forward(fake_images)
            gloss = criterion(out, torch.ones_like(out, dtype=torch.float32, device=device))

            gloss.backward()
            g_batch_acc = torch.sum(torch.round(torch.sigmoid(out))) / batch_size
            g_acc += g_batch_acc

            if True:
                g_optimizer.step()
                skip = True
            else:
                skip = True

            generator_loss += gloss.item()
            discriminator_loss += d_loss.item()

            images.append(draw_image(G, z_fixed))

        g_acc /= len(trainloader)
        generator_loss /= len(trainloader)
        discriminator_loss /= len(trainloader)
        print("Epoch: {} G-Loss: {:.4f} D-Loss: {:.4f} G-Acc: {:.4}".format(epoch, generator_loss,
                                                                            discriminator_loss, g_acc))

    with open('generated_images.pkl', 'wb') as f:
        pickle.dump(images, f)

    torch.save({
                'latent_dim': latent_vector,
                'state_dict': G.state_dict()
               }, "generator.pth")


def draw_image(generator, z):
    generator.eval()
    image = generator.forward(z)
    image = (image + 1) / 2
    image = image.view(16, 32, 32).detach().cpu().numpy()
    generator.train()

    return image


if __name__ == '__main__':
    main()
