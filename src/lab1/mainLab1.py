import src.lab1.hist as hist
import src.lab1.barcode as barcode
import src.lab1.projection as projection
from src.lab1 import utils


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


def main():
    loadHist()
    barcodeTask()
    projectionTask()
