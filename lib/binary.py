import cv2 as cv


def golden_ratio_back(img, reverse=False):
    """
    Делит спектр изображения на две части. Части соотносятся по закону золотого сечения.
    :param img: Исходное изображение;
    :param reverse: Развернуть пропорцию;
    :return: Изображение с разделенным спектром.
    """
    x = img.ravel()
    up = x.mean()/1.618
    if reverse:
        up = x.mean() - up
    _, img2 = cv.threshold(img, up, 255, 0)
    return img2


def puzzles(img, func=golden_ratio_back):
    N = 20
    sh = img.shape
    h, w = sh[0]//N, sh[1]//N
    new_img = img.copy()
    for i in range(N):
        for j in range(N):
            new_img[i*h:(i+1)*h, j*w:(j+1)*w, :] = func(img[i*h:(i+1)*h, j*w:(j+1)*w, :])

    return new_img