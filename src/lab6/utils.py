import cv2 as cv
import matplotlib.pyplot as plt

defaultImagePath = "data/lab2/image.jpg"

applesPath = "data/lab6/apples.jpg"

objectsPath = "data/lab6/objects2.png"

coinsPath = "data/lab6/coins.png"


def loadDefaultImage():
    return cv.imread(defaultImagePath)


def loadApplesImage():
    return cv.imread(applesPath)


def loadObjectsImage():
    return cv.imread(objectsPath)


def loadCoinsImage():
    return cv.imread(coinsPath)


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
