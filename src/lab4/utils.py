import cv2 as cv
import matplotlib.pyplot as plt

defaultImagePath = "data/lab2/image.jpg"

faceImagePath = "data/lab4/face.jpg"

coloredImagePath = "data/lab4/colored.jpg"

texturesPath = "data/lab4/textures2.png"


def loadDefaultImage():
    return cv.imread(defaultImagePath)


def loadFaceImage():
    return cv.imread(faceImagePath)


def loadColoredImage():
    return cv.imread(coloredImagePath)


def loadTextureImage():
    return cv.imread(texturesPath)


def showImage(image, tag, isColored=True):
    if isColored:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    else:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    fig = plt.imshow(image, cmap='gray')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(tag)
    plt.tight_layout()

    path = "result/" + tag.replace(".", ",")
    print(path)
    plt.savefig(path)

    plt.show()
