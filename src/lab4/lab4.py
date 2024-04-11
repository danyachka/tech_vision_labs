import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import src.lab4.utils as utils


def binarize(image: np.ndarray):
    def singleThresholdBinarize(threshold):
        _, result = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
        return result

    def doubleThresholdBinarize(low_threshold, high_threshold):
        _, low = cv.threshold(image, low_threshold, 255, cv.THRESH_BINARY)
        _, high = cv.threshold(image, high_threshold, 255, cv.THRESH_BINARY_INV)
        result = cv.bitwise_and(low, high)
        return result

    def otsuBinarize():
        _, result = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return result

    def adaptiveBinarize(block_size=11, constant=2):
        result = cv.adaptiveThreshold(image, 255,
                                            cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, constant)
        return result

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    single = singleThresholdBinarize(140)
    double = doubleThresholdBinarize(100, 200)
    otsu = otsuBinarize()
    adaptive = adaptiveBinarize(11)

    utils.showImage(single, "Бинаризация")
    utils.showImage(double, "Двойная бинаризация")
    utils.showImage(otsu, "Бинаризация методом Отсу")
    utils.showImage(adaptive, "Адаптивная бинаризация")


def segmentByVeber(image, window_size=15, constant=0.03):
    segmentedImage = np.zeros_like(image)
    height, width, _ = image.shape
    halfWindow = window_size // 2

    for y in range(halfWindow, height - halfWindow):
        for x in range(halfWindow, width - halfWindow):
            # Вычисляем среднее значение интенсивности в окне
            windowMean = np.mean(image[y - halfWindow:y + halfWindow + 1, x - halfWindow:x + halfWindow + 1])

            # Вычисляем порог по методу Вебера
            threshold = windowMean * (1 - constant)

            # Сегментируем пиксель
            if np.all(image[y, x] <= threshold):
                segmentedImage[y, x] = 0
            else:
                segmentedImage[y, x] = 255

    return segmentedImage


def segmentImageKmeansLab(image, k):
    # Преобразование изображения из BGR в CIE Lab
    labImage = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    # Преобразование изображения в одномерный массив для k-средних
    pixelValues = labImage.reshape((-1, 3))

    # Применение k-средних
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixelValues)
    segmentedImage = kmeans.cluster_centers_[kmeans.labels_]

    # Преобразование обратно в формат изображения
    segmentedImage = segmentedImage.reshape(image.shape).astype(np.uint8)

    return segmentedImage


def segmentTexture(image):
    image = cv.imread(utils.texturesPath, cv.IMREAD_GRAYSCALE)

    E = cv.GaussianBlur(image, (5, 5), 0)
    E = cv.convertScaleAbs(E)
    Eim = cv.normalize(E, None, 0, 255, cv.NORM_MINMAX)
    Eim = cv.convertScaleAbs(Eim)

    # Бинаризация энтропии
    _, BW1 = cv.threshold(Eim, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    BWao = cv.morphologyEx(BW1, cv.MORPH_OPEN, np.ones((9, 9), np.uint8))

    # Заливаем чёрным
    Mask1 = cv.bitwise_not(BWao)
    contours, _ = cv.findContours(Mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv.drawContours(Mask1, [contour], 0, 255, -1)

    # Границы
    boundary = cv.Canny(Mask1, 100, 200)
    segmentResults = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    segmentResults[boundary != 0] = (0, 0, 255)

    # Вторая текстура
    I2 = image.copy()
    I2[Mask1 == 255] = 0
    E2 = cv.GaussianBlur(I2, (5, 5), 0)
    E2 = cv.convertScaleAbs(E2)
    E2im = cv.normalize(E2, None, 0, 255, cv.NORM_MINMAX)
    E2im = cv.convertScaleAbs(E2im)

    # бинаризация
    _, BW2 = cv.threshold(E2im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    Mask2 = cv.morphologyEx(BW2, cv.MORPH_OPEN, np.ones((9, 9), np.uint8))

    contours, _ = cv.findContours(Mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv.drawContours(Mask2, [contour], 0, 255, -1)

    # границы 2
    boundary2 = cv.Canny(Mask2, 100, 200)

    # рисуем результат
    texture1 = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    texture1[Mask2 != 255] = 255

    texture2 = image.copy()
    texture2[Mask2 == 255] = 255

    return texture1, segmentResults, texture2, Mask1, Mask2


def printTextureParams(texture, tag):
    #texture = cv.cvtColor(texture, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([texture], [0], None, [256], [0, 256])

    # Обнуляем, так как имеем чёрный фон на месте другой текстуры
    hist[255] = 0

    # Количество всёх "интересных" пикселей
    pixelsCount = int(np.sum(hist))
    print(f"Всего \"интересных\" пикселей в {tag} - {pixelsCount}")
    print(f"Количество всех пикселей в {tag} - {texture.shape[0] * texture.shape[1]}")

    def calcM():
        # Считаем среднее значение z
        m = 0
        for i in range(0, 255):
            m += hist[i][0] * i / pixelsCount
        return m

    def calcMu(n, m):
        mu = 0
        for i in range(0, 255):
            zi = hist[i][0]
            pz = zi / pixelsCount
            mu += (i - m) ** n * pz

        return mu

    M = calcM()
    print(f"m = {M}")

    mu3 = calcMu(3, M)
    #print(f"mu3(z) = {mu3}")

    sigmaSquared = calcMu(2, M)
    s = sigmaSquared ** 0.5
    print(f"s = {s}")

    R = 1 - 1 / (1 + sigmaSquared / 255**2)
    print(f'R = {R}')

    plt.figure(figsize=(10, 6))
    plt.plot(hist, color="blue")
    plt.title("Гистограмма " + tag)
    plt.show()


def main():
    def firstPart():
        image = utils.loadDefaultImage()

        utils.showImage(image, "Оригинальное изображение")
        binarize(image)

    def secondPart():
        faceImage = utils.loadFaceImage()
        veber = segmentByVeber(faceImage)
        utils.showImage(faceImage, "Оригинальное изображение")
        utils.showImage(veber, "Сегментация методом Вебера")

    def thirdPart():
        coloredImage = utils.loadColoredImage()
        result = segmentImageKmeansLab(coloredImage, 5)
        utils.showImage(coloredImage, "Оригинальное изображение")
        utils.showImage(result, "Сегментация по методу k-средних")

    def fourthPart():
        texture = utils.loadTextureImage()
        #utils.showImage(texture, "Оригинальное изображение")

        texture1, segmentResults, texture2, Mask1, Mask2 = segmentTexture(texture)
        utils.showImage(texture1, "Первая текстура")
        # utils.showImage(segmentResults, "Сегментация")
        utils.showImage(texture2, "Вторая текстура")
        # utils.showImage(Mask1, "Первая маска")
        # utils.showImage(Mask2, "Вторая маска")

        printTextureParams(texture1, "(1)")
        print("\n\n")
        printTextureParams(texture2, "(2)")

    fourthPart()


