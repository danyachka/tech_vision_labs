import cv2 as cv
import matplotlib.pyplot as plt

histPath = "data/image.jpg"
hist2Path = "data/lfyz.png"

barcodePath = "data/barcode.jpeg"

projectionImagePath = "data/projection.png"
projection2ImagePath = "data/projection2.png"


def loadHistImage():
    return cv.imread(histPath)


def loadBarcodeRGB():
    image = cv.imread(barcodePath)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def loadProjectionImageRGB():
    image = cv.imread(projection2ImagePath)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def showImage(image, tag):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    fig = plt.imshow(image)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(tag)

    plt.savefig("result/" + tag)

    plt.show()
