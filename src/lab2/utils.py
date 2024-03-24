import cv2 as cv
import matplotlib.pyplot as plt


defaultImagePath = "data/lab2/image.jpg"


def loadDefaultImage():
    return cv.imread(defaultImagePath)


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
