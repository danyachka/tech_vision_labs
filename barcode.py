import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import utils


def showProfile():
    image = utils.loadBarcodeRGB()

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.title("Штрихкод")

    plt.imshow(image)
    plt.axis('off')

    plt.subplot(2, 1, 2)

    level = round(len(image) / 2)

    profile = image[level, :]
    plt.plot(profile, color="black")
    plt.title("Профиль изображения")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.subplots_adjust(left=0.1, right=0.9)

    plt.savefig("result/profile")

    plt.show()

