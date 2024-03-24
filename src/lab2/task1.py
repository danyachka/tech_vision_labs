import cv2 as cv
import src.lab2.utils as utils
import numpy as np


def movePixels(dx, dy):
    image = utils.loadDefaultImage()
    rows, cols = image.shape[0:2]
    T = np.float32([[1, 0, dx], [0, 1, dy]])
    result = cv.warpAffine(image, T, (cols, rows))

    utils.showImage(result, "Сдвинутое изображение")


def mirrorImage(Ox=False, Oy=True):
    image = utils.loadDefaultImage()
    rows, cols = image.shape[0:2]

    label = "Развернутое относительно "
    if Ox and Oy:
        T = np.float32([[-1, 0, cols - 1],
                        [0, -1, rows - 1]])
        label += "Ox и Oy"
    elif Ox:
        T = np.float32([[1, 0, 0],
                        [0, -1, rows - 1]])
        label += "Ox"
    elif Oy:
        T = np.float32([[-1, 0, cols - 1],
                        [0, 1, 0]])
        label += "Oy"
    else:
        T = np.float32([[1, 0, 0],
                        [0, 1, 0]])

    result = cv.warpAffine(image, T, (cols, rows))
    utils.showImage(result, label)


def scale(x=1, y=1):
    image = utils.loadDefaultImage()
    rows, cols = image.shape[0:2]

    label = f"Изменение масштаба (x={x}, y={y})"

    T = np.float32([[x, 0, 0],
                    [0, y, 0]])

    result = cv.warpAffine(image, T, (int(cols * x), int(rows * y)))
    utils.showImage(result, label)


def rotateImage(angle: int = 90):
    image = utils.loadDefaultImage()
    rows, cols = image.shape[0:2]
    rad = angle * np.pi / 180

    label = f"Поворот на {int(angle)}°"

    T = np.float32([[np.cos(rad), -np.sin(rad), 0],
                    [np.sin(rad), np.cos(rad), 0]])
    result = cv.warpAffine(image, T, (cols, rows))
    utils.showImage(result, label)


def rotateImageAroundCenter(angle: int = 90):
    image = utils.loadDefaultImage()
    rows, cols = image.shape[0:2]
    rad = angle * np.pi / 180

    label = f"Поворот на {int(angle)}° около центра изображения"

    T1 = np.float32(
        [[1, 0, -(cols - 1) / 2.0], [0, 1, -(rows - 1) / 2.0], [0, 0, 1]])
    T2 = np.float32(
        [[np.cos(rad), - np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
    T3 = np.float32(
        [[1, 0, (cols - 1) / 2.0], [0, 1, (rows - 1) / 2.0], [0, 0, 1]])

    T = np.matmul(T3, np.matmul(T2, T1))[0:2, :]

    result = cv.warpAffine(image, T, (cols, rows))
    utils.showImage(result, label)


def affineTransform():
    image = utils.loadDefaultImage()
    rows, cols = image.shape[0:2]

    pts_src = np.float32([[50, 300], [150, 200], [50, 50]])
    pts_dst = np.float32([[50, 200], [250, 200], [50, 100]])
    T = cv.getAffineTransform(pts_src, pts_dst)

    result = cv.warpAffine(image, T, (cols, rows))
    utils.showImage(result, "Аффинное отображение")


def bevelImage(s=0.3):
    image = utils.loadDefaultImage()
    rows, cols = image.shape[0:2]

    T = np.float32([[1, s, 0], [0, 1, 0]])
    result = cv.warpAffine(image, T, (cols, rows))

    utils.showImage(result, "Скос изображения")


def piecewiseLinearTransform(sLeft=0.5, sRight=2):
    image = utils.loadDefaultImage()
    rows, cols = image.shape[0:2]

    result = image.copy()

    TLeft = np.float32([[sLeft, 0, 0], [0, 1, 0]])
    result[:, :int(cols / 2), :] = (
        cv.warpAffine(result[:, :int(cols / 2), :], TLeft, (cols - int(cols / 2), rows)))

    rightPart = int(sLeft * cols / 2)
    TRight = np.float32([[sRight, 0, 0], [0, 1, 0]])
    result[:, rightPart:, :] = (
        cv.warpAffine(result[:, int(cols / 2):, :], TRight, (cols - rightPart, rows)))

    utils.showImage(result, "Кусочно-линейное преобразование")


# Значения букв 4 строчки ниже (стоит подписать в отчёте)
def projectiveTransform():
    image = utils.loadDefaultImage()
    rows, cols = image.shape[0:2]

    ABC = [0.9, 0.15, 0.00075]
    DEF = [0.29, 1.4, 0.0004]
    GH1 = [0, 0, 1]
    T = np.float32([ABC, DEF, GH1])

    result = cv.warpPerspective(image, T, (cols, rows))

    utils.showImage(result, "Проекционное преобразование")


def sinTransform():
    image = utils.loadDefaultImage()
    rows, cols = image.shape[0:2]

    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    u = u + 80 * np.sin(0.5 * np.pi * v / 90)
    result = cv.remap(image, u.astype(np.float32), v.astype(np.float32), cv.INTER_LINEAR)

    utils.showImage(result, "Синусоидальное преобразование")
