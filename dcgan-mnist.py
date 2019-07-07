import click
import torchvision.datasets.mnist as mnist
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import torch


@click.command()
@click.option('--root', default="~/.torch/mnist", help="Root directory for MNIST dataset")
@click.option('--batch-size', default=128, help="Batch size")
def main(root, batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x * 2 - 1)])
    trainset = mnist.MNIST(root=root, download=True, train=True, transform=transform)
    testset = mnist.MNIST(root=root, download=True, train=False, transform=transform)
    trainloader = dataloader.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = dataloader.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, drop_last=True)

    for x_image, y_label in trainloader:
        pass


if __name__ == '__main__':
    main()
