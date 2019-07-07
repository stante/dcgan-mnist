import click
import torchvision.datasets.mnist as mnist
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import torch


@click.command()
@click.option('--root', default="~/.torch/mnist", help="Root directory for MNIST dataset")
@click.option('--batch-size', default=128, help="Batch size")
@click.option('--latent-vector', default=100, help="Size of latent vector Z")
@click.option('--disable-cuda', default=False, help="Disable CUDA acceleration")
def main(root, batch_size, latent_vector, disable_cuda):
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x * 2 - 1)])
    trainset = mnist.MNIST(root=root, download=True, train=True, transform=transform)
    testset = mnist.MNIST(root=root, download=True, train=False, transform=transform)
    trainloader = dataloader.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = dataloader.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, drop_last=True)

    for x_image, y_label in trainloader:
        x_image = x_image.to(device)
        y_label = y_label.to(device)

        z = torch.rand((latent_vector, 1), device=device) * 2 - 1
        pass


if __name__ == '__main__':
    main()
