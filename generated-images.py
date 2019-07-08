import matplotlib.pyplot as plt
import pickle


def main():
    with open('generated_images_01.pkl', 'rb') as f:
        images = pickle.load(f)

    for n, image in enumerate(images):
        fig = plt.figure()

        for i in range(16):
            fig.add_subplot(4, 4, i + 1)
            plt.imshow(image[i], cmap='gray')

        plt.savefig("data/fig_{:03}.png".format(n))


if __name__ == '__main__':
    main()
