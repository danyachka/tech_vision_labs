import src.lab2.task1 as task1


def firstTask():
    # shift
    task1.movePixels(20, -30)

    # mirror
    task1.mirrorImage(Oy=True)
    task1.mirrorImage(Ox=True, Oy=True)

    # scale
    task1.scale(4, 2)

    # rotate
    task1.rotateImage(45)
    task1.rotateImageAroundCenter(45)

    # Дальше Аффинное, стр 39
    task1.affineTransform()


def main():
    firstTask()
