import cv2 as cv


def balance_brightness(img):
    """
    Исправление яркости изображения.
    :param img: исходное изображение;
    :return: Изображение с исправленной яркостью
    """
    shape = img.shape
    h, w = shape[0], shape[1]
    yar0 = img.sum() / (w * h * 3 * 255)
    yar = (0.5 + abs(0.5 - yar0)) * 0.95
    img_ = cv.filter2D(img, -1, 1/yar)
    return img_
