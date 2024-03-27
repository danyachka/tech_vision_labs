import cv2 as cv
import numpy as np
import src.lab2.utils as utils


def getBarrel(image) -> np.ndarray:
    rows, cols = image.shape[0:2]

    xi, yi = np.meshgrid(np.arange(cols), np.arange(rows))
    xMid = cols / 2.0
    yMid = rows / 2.0

    xi = xi - xMid
    yi = yi - yMid

    r, theta = cv.cartToPolar(xi / xMid, yi / yMid)
    F3 = 0.1
    F5 = 0.12
    r = r + F3 * r ** 3 + F5 * r ** 5

    u, v = cv.polarToCart(r, theta)
    u = u * xMid + xMid
    v = v * yMid + yMid
    result = cv.remap(image, u.astype(np.float32), v.astype(np.float32), cv.INTER_LINEAR)

    utils.showImage(result, "Бочкообразная дисторсия")
    return result


def fixBarrel(image, F3, F5):
    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

    x, y = (map_x - w / 2) / (w / 2), (map_y - h / 2) / (h / 2)
    r2 = x ** 2 + y ** 2
    r4 = r2 ** 2
    r6 = r2 * r4

    radial_coeff = 1 + F3 * r2 + F5 * r4

    x_distorted = x * radial_coeff
    y_distorted = y * radial_coeff

    map_x_distorted = ((x_distorted * (w / 2)) + w / 2).astype(np.float32)
    map_y_distorted = ((y_distorted * (h / 2)) + h / 2).astype(np.float32)

    corrected_img = cv.remap(image, map_x_distorted, map_y_distorted, cv.INTER_LINEAR)

    utils.showImage(corrected_img, "Исправленное (бочкообразная дисторсия)")


def getPincushion(image) -> np.ndarray:
    rows, cols = image.shape[0:2]

    xi, yi = np.meshgrid(np.arange(cols), np.arange(rows))
    xMid = cols / 2.0
    yMid = rows / 2.0

    xi = xi - xMid
    yi = yi - yMid

    r, theta = cv.cartToPolar(xi / xMid, yi / yMid)
    F3 = -0.003
    F5 = -0.12
    r = r + F3 * r ** 3 + F5 * r ** 5

    u, v = cv.polarToCart(r, theta)
    u = u * xMid + xMid
    v = v * yMid + yMid
    result = cv.remap(image, u.astype(np.float32), v.astype(np.float32), cv.INTER_LINEAR)

    utils.showImage(result, "Подушкообразная дисторсия")
    return result


def fixPincushion(image, F3, F5):
    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

    x, y = (map_x - w / 2) / (w / 2), (map_y - h / 2) / (h / 2)
    r2 = x ** 2 + y ** 2
    r4 = r2 ** 2
    r6 = r2 * r4

    radial_coeff = 1 + F3 * r2 + F5 * r4

    x_distorted = x * radial_coeff
    y_distorted = y * radial_coeff

    map_x_distorted = ((x_distorted * (w / 2)) + w / 2).astype(np.float32)
    map_y_distorted = ((y_distorted * (h / 2)) + h / 2).astype(np.float32)

    corrected_img = cv.remap(image, map_x_distorted, map_y_distorted, cv.INTER_LINEAR)

    utils.showImage(corrected_img, "Исправленное (подушкообразная дисторсия)")
