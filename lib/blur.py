import cv2 as cv
import numpy as np


def subtracting_white_spots(img, wt=0.75, md=17):  # ПЕРВАЯ ЭВРИСТИКА
    """
    Происходит размытие фона медийным фильтром. Получаем светлые и темные пятна.
    С определенным весом отнимается из исходного изображения полученные пятна,
    темные пятна не влияют (малые значения), а вот белые оказывают влияние.
    Как правило, текст темные пятна, а фон светлые.
    :param img: Изображение у которого необходимо убрать фон;
    :param wt: Вес фона [0-1] рекомендуемое значение 0.75;
    :param md: Медиана размытия, рекомендуется 17 пикселей;
    :return: Исправленное изображение
    """

    img = (img - cv.medianBlur(img, md) * wt)
    low_background = (1, 1, 1)
    high_background = (255, 255, 255)
    only_text = cv.inRange(img, low_background, high_background)

    return only_text


def sharpening(img):
    """
    НАДО ОПИСАТЬ
    :param img: исходное изображение;
    :return: изображение с повышенной резкостью
    """
    gaussian = cv.GaussianBlur(img, (0, 0), 2.0)
    after_img = cv.addWeighted(img, 2.0, gaussian, -1.0, 0)
    return after_img


def convolution_method(img, kernel=None):
    """
    Свертка. Значение пикселя изображения зависит также от соседних пикселей.
    :param img: Исходное изображение;
    :return: После свертки.
    """
    if kernel is None:
        kernel = np.array([[1, 0, -1],
                          [1, 6, 1],
                          [1, 0, -1]])

    kernel = 1/(kernel.sum())*kernel

    img2 = cv.filter2D(img, -1, kernel)

    return img2

