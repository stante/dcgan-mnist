import click
import torchvision.datasets.mnist as mnist
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import torch


@click.command()
def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x * 2 - 1)])
    trainset = mnist.MNIST(root="~/.torch/mnist", download=True, train=True, transform=transform)
    testset = mnist.MNIST(root="~/.torch/mnist", download=True, train=False, transform=transform)
    trainloader = dataloader.DataLoader(dataset=trainset, batch_size=128, shuffle=True, drop_last=True)
    testloader = dataloader.DataLoader(dataset=testset, batch_size=128, shuffle=True, drop_last=True)

    for x_image, y_label in trainloader:
        pass


if __name__ == '__main__':
    main()
