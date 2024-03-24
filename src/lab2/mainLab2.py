import src.lab2.task1 as task1


def firstTask():

    def linear():
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

        # Аффинное
        task1.affineTransform()

    def nonLinear():
        # Скос
        task1.bevelImage()

        # кусочно-линейное
        task1.piecewiseLinearTransform(0.3, 3)

        # Проекционное
        task1.projectiveTransform()

        # Синусоидальное
        task1.sinTransform()

    #linear()
    nonLinear()


def main():
    firstTask()
