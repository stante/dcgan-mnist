import click
import torchvision.datasets.mnist as mnist
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn as nn
import model


@click.command()
@click.option('--root', default="~/.torch/mnist", help="Root directory for MNIST dataset")
@click.option('--epochs', default=10, help="Number of epochs")
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

    generator = model.DCGANModelGenerator(latent_vector)
    discriminator = model.DCGANModelDiscriminator()
    gen_optim = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    disc_optim = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.MSELoss()
    generator.to(device)
    discriminator.to(device)

    for epoch in range(1, epochs + 1):
        generator_loss = 0
        discriminator_loss = 0
        for x_image, y_label in trainloader:
            x_image = x_image.to(device)
            # y_label = y_label.to(device)

            discriminator.train()
            generator.eval()

            disc_optim.zero_grad()
            z = torch.rand((batch_size, latent_vector), device=device) * 2 - 1
            x_gen = generator.forward(z)
            out = discriminator.forward(torch.cat([x_image, x_gen]))
            dloss = criterion(out, torch.cat((torch.ones_like(y_label, dtype=torch.float32, device=device),
                                              torch.zeros_like(y_label, dtype=torch.float32, device=device))))
            dloss.backward()
            disc_optim.step()

            discriminator.eval()
            generator.train()

            gen_optim.zero_grad()
            z = torch.rand((batch_size, latent_vector), device=device) * 2 - 1
            x_gen = generator.forward(z)
            out = discriminator.forward(x_gen)
            gloss = criterion(out, torch.ones_like(out, dtype=torch.float32, device=device))

            gloss.backward()
            gen_optim.step()

            generator_loss += gloss.item()
            discriminator_loss += dloss.item()

        generator_loss /= len(trainloader)
        discriminator_loss /= len(trainloader)
        print("Epoch: {} G-Loss: {:.4f} D-Loss: {:.4f}".format(epoch, generator_loss, discriminator_loss))


if __name__ == '__main__':
    main()
