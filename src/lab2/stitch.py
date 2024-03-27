import cv2 as cv
import numpy as np
import src.lab2.utils as utils


def stitch():
    top = utils.loadTop()
    bottom = utils.loadBottom()

    templ_size = 10
    templ = top[- templ_size:, :, :]

    res = cv.matchTemplate(bottom, templ, cv.TM_CCOEFF)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    result = np.zeros((top.shape[0] + bottom.shape[0] - max_loc[1] - templ_size, top.shape[1], top.shape[2]),
                      dtype=np.uint8)

    h, w = result.shape[:2]
    print(f"{h}:{w}")

    h, w = top.shape[:2]
    print(f"{h}:{w}")

    h, w = bottom.shape[:2]
    print(f"{h}:{w}")

    result[0: top.shape[0], :, :] = top
    result[top.shape[0]:, :, :] = bottom[max_loc[1] + templ_size:, :, :]

    utils.showImage(result, "Сшитое изображение")

