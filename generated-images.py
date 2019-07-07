import matplotlib.pyplot as plt
import pickle


with open('generated_images.pkl', 'rb') as f:
    images = pickle.load(f)

for image in images:
    fig = plt.figure()

    for i in range(16):
        fig.add_subplot(4, 4, i + 1)
        plt.imshow(image[i], cmap='gray')

    plt.show()
