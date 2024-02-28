import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import utils


def createHist():
    b, g, r = cv.split(utils.loadHistImage())

    blueHist = cv.calcHist([b], [0], None, [256], [0, 256])
    plt.figure(figsize=(10, 6))
    plt.subplot(311)
    plt.plot(blueHist, color="blue")
    plt.title("Blue hist")

    greenHist = cv.calcHist([g], [0], None, [256], [0, 256])
    plt.subplot(312)
    plt.plot(greenHist, color="green")
    plt.title("Green hist")

    redHist = cv.calcHist([r], [0], None, [256], [0, 256])
    plt.subplot(313)
    plt.plot(redHist, color="red")
    plt.title("Red hist")

    plt.tight_layout()

    plt.savefig("result/hist")
    plt.show()


def equalizeHistogram():
    image = utils.loadHistImage()
    blue, green, red = cv.split(image)

    newB = cv.equalizeHist(blue)
    newG = cv.equalizeHist(green)
    newR = cv.equalizeHist(red)

    equalizedImage = cv.merge([newB, newG, newR])

    utils.showImage(equalizedImage, "Выравненное изображение")


def equalizeHistogramWithStretching():
    blue, green, red = cv.split(utils.loadHistImage())

    newB = cv.equalizeHist(blue)
    newG = cv.equalizeHist(green)
    newR = cv.equalizeHist(red)

    equalizedImage = cv.merge([newB, newG, newR])

    param1, param2 = np.percentile(equalizedImage, (5, 95))
    result = cv.normalize(equalizedImage, None, param1, param2, cv.NORM_MINMAX)
    utils.showImage(result, "Выравненное с растянутым контрастом")


def nonlinearStretching(alpha):
    image = utils.loadHistImage()
    needConvert: bool = image.dtype == np.uint8
    if needConvert:
        image = image.astype(np.float32) / 255

    blue, green, red = cv.split(utils.loadHistImage())

    minBlue = blue.min()
    maxBlue = blue.max()

    minGreen = green.min()
    maxGreen = green.max()

    minRed = red.min()
    maxRed = red.max()

    newBlue = np.clip(((blue - minBlue) / (maxBlue - minBlue)) ** alpha, 0, 1)
    newGreen = np.clip(((green - minGreen) / (maxGreen - minGreen)) ** alpha, 0, 1)
    newRed = np.clip(((red - minRed) / (maxRed - minRed)) ** alpha, 0, 1)

    result = cv.merge([newBlue, newGreen, newRed])
    if needConvert:
        result = result * 255
        result = result.clip(0, 255).astype(np.uint8)

    utils.showImage(result, "Нелинейное растяжение")


def linearStretching():
    image = utils.loadHistImage()
    hist = cv.calcHist([image], [0], None, [256], [0, 256])

    cumHist = np.cumsum(hist) / (image.shape[0] * image.shape[1])

    minI = np.min(image)
    maxI = np.max(image)

    result = np.round(minI + (maxI - minI) * cumHist[image])
    result = result.astype(np.uint8)

    utils.showImage(result, "Линейное растяжение")


def exponentialTransform(alpha):
    image = utils.loadHistImage()
    image = image.astype(np.float32) / 255

    imageExp = np.power(image, alpha) * 255

    imageExp = np.clip(imageExp, 0, 255).astype(np.uint8)
    utils.showImage(imageExp, "Экспоненциальное растяжение")


def _get_normalized_hist(image):
    divider = image.shape[0] * image.shape[1]
    hist = []
    for i in range(3):
        hist.append(cv.calcHist([image], [i], None, [256], (0, 256)) / divider)
    return hist


def rayleighTransform(alpha):
    image = utils.loadHistImage()

    hist = _get_normalized_hist(image)

    layers = cv.split(image)

    resultHist = []
    for i in range(3):
        layer = layers[i]

        maxL, minL = layer.max(), layer.min()
        cumulativeHist = np.cumsum(hist[i])

        c = cumulativeHist[layer]

        resultLayer = np.clip(minL + 255 * (2*alpha**2 * np.log(1 / (1 - c))) ** 0.5, 0, 255)
        resultHist.append(resultLayer)

    result = cv.merge(resultHist).astype(np.uint8).clip(0, 255)
    utils.showImage(result, "Преобразование по закону Рэлея")


# По закону степени 2/3
def expTransformByRule():
    image = utils.loadHistImage()

    hist = _get_normalized_hist(image)

    layers = cv.split(image)

    result = []
    for i in range(3):
        cumulativeHist = np.cumsum(hist[i])

        layer = layers[i]

        resultLayer = np.clip(255 * (cumulativeHist[layer] ** (2/3)), 0, 255)
        result.append(resultLayer)

    resultImage = cv.merge(result).astype(np.uint8)
    utils.showImage(resultImage, "Преобразование по закону степени две трети")


def hypImage(alpha, beta):
    image = utils.loadHistImage().astype(np.float32) / 255

    newImage = np.arcsinh(image) * 255

    result = cv.convertScaleAbs(newImage, alpha, beta)

    utils.showImage(result, "Гиперболическое преобразование")
