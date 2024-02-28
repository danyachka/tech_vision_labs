import hist
import barcode
import projection
import utils


def loadHist():
    # show default image
    utils.showImage(utils.loadHistImage(), "Начальное изображение")
    hist.createHist()
    hist.equalizeHistogram()
    hist.equalizeHistogramWithStretching()
    hist.nonlinearStretching(0.7)
    hist.linearStretching()
    hist.exponentialTransform(1.4)
    hist.rayleighTransform(0.4)
    hist.expTransformByRule()
    hist.hypImage(1.5, 0.7)


def barcodeTask():
    barcode.showProfile()


def projectionTask():
    projection.showProjection()


if __name__ == '__main__':
    loadHist()
    barcodeTask()
    projectionTask()
