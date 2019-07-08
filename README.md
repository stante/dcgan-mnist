# dcgan-mnist
This repository contains an implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) for MNIST.
It implements the suggested architectural constraints for stable learning from the paper
[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
by *Alec Radford, Luke Metz, Soumith Chintala*.

The generator and discriminator are all convolutional networks without max pooling but instead uses strided (and 
fractionally strided) convolutions. Fruthermore, batch normalization is used and ReLu and Leaky ReLU activations.

## Usage
```sh
Usage: dcgan-mnist.py [OPTIONS]

Options:
  --root TEXT              Root directory for MNIST dataset
  --epochs INTEGER         Number of epochs
  --batch-size INTEGER     Batch size
  --latent-vector INTEGER  Size of latent vector Z
  --disable-cuda TEXT      Disable CUDA acceleration
  --help                   Show this message and exit.
```


## License
dcgan-mnist is Copyright © 2019 Alexander Stante. It is free software, and may be redistributed under the 
terms specified in the [LICENSE](/LICENSE) file.