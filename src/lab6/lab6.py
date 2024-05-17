import src.lab6.utils as utils
import cv2 as cv
import numpy as np


def bwareaopen(img, A, dim, conn=8):
    if img.ndim > 2:
        return None
    # Find all connected components
    num, labels, stats, centers = cv.connectedComponentsWithStats(A, connectivity=conn)
    # Check size of all connected components
    for i in range(num):
        if stats[i, cv.CC_STAT_AREA] < dim:
            A[labels == i] = 0
    return A


def morphologicalOperations():
    image = utils.loadApplesImage()
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    _, binary = cv.threshold(image, 110, 255, cv.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    eroded = cv.erode(binary, kernel, iterations=2)

    dilated = cv.dilate(eroded, kernel, iterations=2)

    utils.showImage(image, "Оригинальное изображение")
    utils.showImage(eroded, "Результат наложения эрозии")
    utils.showImage(dilated, "Результат наложения дилатации")


def borderObjects():
    image = utils.loadObjectsImage()
    imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    _, binary = cv.threshold(imageGray, 200, 255, cv.THRESH_BINARY_INV)

    struct = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    eroded = cv.morphologyEx(binary, cv.MORPH_ERODE, struct, iterations=12,
                             borderType=cv.BORDER_CONSTANT,
                             borderValue=[0])

    result = np.zeros_like(binary)
    while cv.countNonZero(eroded) < eroded.size:
        dilated = cv.dilate(eroded, struct,
                            borderType=cv.BORDER_CONSTANT,
                            borderValue=[0])
        closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, struct,
                                 borderType=cv.BORDER_CONSTANT,
                                 borderValue=[0])

        result = cv.bitwise_or(closed - dilated, result)
        eroded = dilated

    result = cv.morphologyEx(result, cv.MORPH_CLOSE, struct, iterations=12,
                             borderType=cv.BORDER_CONSTANT,
                             borderValue=[255])

    objects = cv.bitwise_and(~result, binary)
    utils.showImage(image, "Оригинальное изображение")
    utils.showImage(binary, "Бинарное изображение")
    utils.showImage(objects, "Результат")
    utils.showImage(result, "Границы")

    contours, _ = cv.findContours(objects, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0, 255, 255), 6)
    utils.showImage(image, "Оригинальное изображение с границами")


def segmentImage():
    image = utils.loadCoinsImage()
    resultImage = image.copy()
    imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    imageGray = cv.GaussianBlur(imageGray, [15, 15], 0)

    _, binary = cv.threshold(imageGray, 170, 255,
                             cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    utils.showImage(binary, "Бинарное изображение")
    binary = bwareaopen(binary, binary, 20, 4)

    struct = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    opened = cv.morphologyEx(binary, cv.MORPH_CLOSE, struct)

    foreground = cv.distanceTransform(opened, cv.DIST_L2, 5)
    _, foreground = cv.threshold(foreground, 0.3 * foreground.max(), 255, 0)
    foreground = (foreground * 255).astype(np.uint8)
    print(foreground.dtype)
    num, markers = cv.connectedComponents(foreground)

    # Back
    background = np.zeros_like(opened)
    markersBackground = markers.copy()
    markersBackground = cv.watershed(image, markersBackground)
    background[markersBackground == -1] = 255

    unknown = cv.subtract(~background, foreground)

    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv.watershed(image, markers)
    markers = (markers.astype(np.float32) * 255 / num + 1).astype(np.uint8)
    markersJet = cv.applyColorMap(markers, cv.COLORMAP_JET)

    canny = np.uint8(cv.Canny(markers, 60, 160))
    contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(resultImage, contours, -1, (0, 255, 255), 6)

    utils.showImage(image, "Оригинальное изображение")
    utils.showImage(markersJet, "В пространстве Jet")
    utils.showImage(markers, "Маркеры")
    utils.showImage(resultImage, "Границы")


def main():
    segmentImage()


if __name__ == '__main__':
    main()
