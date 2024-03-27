import cv2 as cv
import matplotlib.pyplot as plt


defaultImagePath = "data/lab2/image.jpg"

barrelImage = "data/lab2/barrelDistortionImage.jpg"

topImage = "data/lab2/top.jpg"
bottomImage = "data/lab2/bottom.jpg"


def loadDefaultImage():
    return cv.imread(defaultImagePath)


def loadBarrel():
    return cv.imread(barrelImage)


def loadTop():
    return cv.imread(topImage)


def loadBottom():
    return cv.imread(bottomImage)


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
