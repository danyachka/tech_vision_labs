import cv2 as cv
import matplotlib.pyplot as plt

defaultImagePath = "data/lab2/image.jpg"

barcode = "data/lab5/barcode.jpeg"

lines = "data/lab5/lines.png"

road = "data/lab5/road.jpeg"

circles = "data/lab5/circles.png"

bubbles = "data/lab5/bubbles.jpg"

car = "data/lab5/car.png"


def loadDefaultImage():
    return cv.imread(defaultImagePath)


def loadBarcodeImage():
    return cv.imread(barcode)


def loadLinesImage():
    return cv.imread(lines)


def loadRoadImage():
    return cv.imread(road)


def loadCirclesImage():
    return cv.imread(circles)


def loadBubblesImage():
    return cv.imread(bubbles)


def loadCar():
    return cv.imread(car)


def showImage(image, tag, isColored=True):
    if isColored:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    else:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # plt.figure(figsize=(9, 6))
    fig = plt.imshow(image, cmap='gray')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(tag)
    plt.tight_layout()

    path = "result/" + tag.replace(".", ",")
    print(path)
    plt.savefig(path)

    plt.tight_layout()
    plt.show()
