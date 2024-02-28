import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import utils


def showProjection():
    image = utils.loadProjectionImageRGB().astype(np.float64)

    layers = cv.split(image)

    xLen = image.shape[0]
    yLen = image.shape[1]

    xProjection = None
    yProjection = None
    for i in range(3):
        layer = layers[i]
        if i == 0:
            xProjection = np.sum(layer, axis=0)
            yProjection = np.sum(layer, axis=1)
        else:
            xProjection += np.sum(layer, axis=0)
            yProjection += np.sum(layer, axis=1)

    xProjection /= (3 * xLen)
    yProjection /= (3 * yLen)

    xProjection = xProjection
    yProjection = np.flip(yProjection)

    # Исходное изображение
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(2, 2, 1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.set_title("Изображение")
    plt.imshow(image.astype(np.uint8))

    # y
    ax_y = plt.subplot(2, 2, 2)
    ax_y.set_xlim(0, 255)
    ax_y.set_ylim(0, xLen)
    plt.grid(True)
    ax_y.plot(yProjection, range(xLen), color="blue")
    ax_y.set_title("Проекция y")

    # x
    ax_x = plt.subplot(2, 2, 3)
    ax_x.set_xlim(0, yLen)
    ax_x.set_ylim(0, 255)
    plt.grid(True)
    ax_x.plot(range(yLen), xProjection, color="blue")
    ax_x.set_title("Проекция x")

    plt.savefig("result/projections")
    plt.show()

