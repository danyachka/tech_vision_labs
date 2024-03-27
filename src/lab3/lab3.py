import numpy as np
import cv2 as cv
import src.lab3.utils as utils
import src.lab3.filters as filters


def secondTask(noised, level):
    gaussDenoise = cv.GaussianBlur(noised, (9, 9), 0)
    utils.showImage(gaussDenoise, f"Gaussian denoise(noise level={level})")

    for q in [1, -1.5, -3]:
        denoise = filters.contraharmonicMeanFilter(noised, 3, q)
        utils.showImage(denoise, f"Contraharmonic Mean Filter(noise level={level}, Q={q})")


def thirdTask(noised, level):
    median = filters.medianFilter(noised, 9)
    utils.showImage(median, f"Медианный фильтр (noise level={level})")

    weightedMedian = filters.weightedMedianFilter(noised, 3)
    utils.showImage(weightedMedian, f"Взвешенный медианный фильтр (noise level={level})")

    rank = 4
    rankImage = filters.rankFilter(noised, 7, rank)
    utils.showImage(rankImage, f"Ранговый фильтр (noise level={level}, rank={rank})")

    for mask, noise_var in [(7, 100), (5, 50), (5, 250)]:
        weinerImage = filters.weinerFilter(noised, mask, noise_var)
        utils.showImage(weinerImage,
                        f"Винеровский фильтр (noise level={level}, noise_var={noise_var}, mask={mask})")

    adaptiveMedian = filters.adaptiveMedianFilter(noised, 5)
    utils.showImage(adaptiveMedian, f"Адаптивная медианная фильтрация (noise level={level})")


def main():
    image = utils.loadDefaultImage()
    utils.showImage(image, "Оригинал")

    # for level in [0.7]:
    for level in [0.4, 0.6, 0.7]:
        noised = filters.createGaussianNoise(image, level)

        utils.showImage(noised, f"Шум (noise level={level})")

        # secondTask(noised, level)

        thirdTask(noised, level)

