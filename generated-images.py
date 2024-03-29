import matplotlib.pyplot as plt
import pickle
import click
import os.path
import math


@click.command()
@click.argument('in-file')
@click.argument('out-dir')
def main(in_file, out_dir):
    with open(in_file, 'rb') as f:
        images = pickle.load(f)

    print("Writing {} images to disk.".format(len(images)))

    num_digits = int(math.log(len(images), 10)) + 1
    for n, image in enumerate(images[::100]):
        fig = plt.figure()

        for i in range(16):
            fig.add_subplot(4, 4, i + 1)
            plt.imshow(image[i], cmap='gray')

        plt.savefig(os.path.join(out_dir, "fig_{:0{}}.png".format(n, num_digits)))


if __name__ == '__main__':
    main()
