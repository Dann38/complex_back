import cv2 as cv
import numpy as np


def balance_brightness(img):
    """
    Исправление яркости изображения.
    :param img: исходное изображение;
    :return: Изображение с исправленной яркостью
    """
    shape = img.shape
    h, w = shape[0], shape[1]
    yar0 = img.sum() / (w * h * 3 * 255)
    # yar = 1 / (0.5 + abs(0.5 - yar0)) / 0.95
    yar = 1 / (0.3 + abs(0.3 - yar0)) * 1.1  # Фантастический результат в 2.5 раза
    img_ = cv.addWeighted(img, yar, img, 0, 0)
    return img_
