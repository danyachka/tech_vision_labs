import cv2 as cv
import matplotlib.pyplot as plt

histPath = "data/lab1/image.jpg"
hist2Path = "data/lab1/lfyz.png"

barcodePath = "data/lab1/barcode.jpeg"

projectionImagePath = "data/lab1/projection.png"
projection2ImagePath = "data/lab1/projection2.png"


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
    plt.tight_layout()

    path = "result/" + tag
    print(path)
    plt.savefig(path)

    plt.show()
