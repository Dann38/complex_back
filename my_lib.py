import random

import cv2 as cv
import pytesseract
import difflib
import numpy as np
import matplotlib.pyplot as plt


def read_img(kyrillic_name):
    with open(kyrillic_name, "rb") as f:
        chunk = f.read()
    chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
    img = cv.imdecode(chunk_arr, cv.IMREAD_COLOR)
    return img


def get_without_back(img, wt=0.75, md=17):
    """
    :param img: Изображение у которого необходимо убрать фон
    :param wt: Вес фона [0-1] рекомендуемое значение 0.8
    :param md: Медиана размытия, рекомендуется 17 пикселей
    :return: Исправленное изображение
    """
    # img_ = (img - cv.medianBlur(img, 31)*wt)
    img_ = (img - cv.medianBlur(img, md) * wt)
    low_background = (1, 1, 1)
    high_background = (255, 255, 255)
    only_text = cv.inRange(img_, low_background, high_background)
    # ====================== Выделить белые строки ===============================
    # for i in range(len(only_text.sum(1))):
    #     if only_text.sum(1)[i] > only_text.sum(1).max()*0.80 :
    #         only_text[i] = 255
    return only_text


def get_without_back2(img):
    # img2 = cv.medianBlur(img2, 3)
    img2 = img.filter(cv.ImageFilter.UnsharpMask(radius=2, percent=150))
    return img2


def get_without_back3(img):
    shape = img.shape
    h, w = shape[0], shape[1]
    yar = img.sum() / (w * h * 3 * 255)
    yar = (0.5 + abs(0.5 - yar))*0.95

    print(yar)
    kernel = np.array([[1, 0, -1],
                       [1, 6, 1],
                       [1, 0, -1]])

    kernel = 1/(8*yar)*kernel

    img2 = cv.filter2D(img, -1, kernel)

    return img2


def get_without_back4(img):
    # img2 = cv.blur(img, (5,5))
    x = img.ravel()
    up = x.mean()/1.618
    # print(up)
    # img_array = img.sum(axis=2)
    _, img2 = cv.threshold(img, up, 255, 0)
    # plt.plot(img2)
    # plt.show()
    # img2 = cv.blur(img2, (3,3))
    return img + img2


def get_text_from_img(img, leng='rus+eng', oem='3', psm='4'):
    """
    :param leng - язык
        0. rus
    :param oem - какой движок использовать
        0. Устаревший движок
        1. Нейронная сеть LSTM
        2. Устаревший движок и LSTM
        3. По умолчанию

    :param psm - сегментация страницы
        0.  Только ориентация и обнаружение сценариев (экранное меню).
        1.  Автоматическая сегментация страницы с помощью экранного меню.
        2.  Автоматическая сегментация страницы, но без экранного меню или распознавания текста. (не реализовано)
        3.  Полностью автоматическая сегментация страниц, но без экранного меню. (По умолчанию)
        4.  Предположим, что один столбец текста имеет переменные размеры.
        5.  Предположим, что это один однородный блок текста, выровненный по вертикали.
        6.  Предположим, что это один однородный блок текста.
        7.  Обрабатывайте изображение как одну текстовую строку.
        8.  Рассматривайте изображение как одно слово.
        9.  Рассматривайте изображение как одно слово в круге.
        10. Обрабатывайте изображение как один символ.
        11. Редкий текст. Найдите как можно больше текста в произвольном порядке.
        12. Разреженный текст с экранным меню.
        13. Необработанная линия. Обрабатывайте изображение как одну текстовую строку,
        минуя хаки, специфичные для Тессеракта.
    :return: text - возвращает текст
    """
    custom_config = f'-l {leng} --oem {oem} --psm {psm}'
    text = pytesseract.image_to_string(img, config=custom_config)
    return text


def similarity(s1, s2):
    normalized1 = s1.lower()
    normalized2 = s2.lower()
    matcher = difflib.SequenceMatcher(None, normalized1, normalized2)
    return matcher.ratio()


def get_img_selected_text(img, leng='rus+eng', oem='3', psm='4'):
    shape = img.shape
    h, w = shape[0], shape[1]
    custom_config = f'-l {leng} --oem {oem} --psm {psm}'

    boxes = pytesseract.image_to_boxes(img, config=custom_config)

    for b in boxes.splitlines():
        b = b.split()
        # cv.rectangle(img, ((int(b[1]), h - int(b[2]))), ((int(b[3]), h - int(b[4]))), (0, 255, 0), 2)

    return img


def remove_frame(img, frame_prc=0.1):
    sh = img.shape
    h, w = sh[0], sh[1]

    # # Получаем массивы диагоналей
    # C = h/w
    # f = lambda x: round(C*x)
    # # print(sh)
    # if len(sh) == 2:
    #     array_1 = np.array([img[f(xi), xi] for xi in range(w)])
    #     array_2 = np.array([img[f(xi), (w - 1) - xi] for xi in range(w)])
    #
    #     array_3 = np.array([img[round(xi * C), round(w / 2)] for xi in range(w)])
    #     array_4 = np.array([img[round(h / 2), xi] for xi in range(w)])
    # else:
    #     array_1 = np.array([img[f(xi), xi, :].sum() for xi in range(w)])
    #     array_2 = np.array([img[f(xi), (w-1)-xi,].sum() for xi in range(w)])
    #
    #     array_3 = np.array([img[round(xi*C), round(w/2), :].sum() for xi in range(w)])
    #     array_4 = np.array([img[round(h/2), xi, :].sum() for xi in range(w)])

    img_array = img.sum(axis=2)/(255*3)

    h_stripe = [round(h/2 - h*0.05), round(h/2 + h*0.05)]
    w_stripe = [round(w/4 - w*0.05), round(w/4 + w*0.05)]
    stripe_img_w = img_array[h_stripe[0]:h_stripe[1], :]
    stripe_img_h = img_array[:, w_stripe[0]:w_stripe[1]]

    array_w = stripe_img_w.sum(axis=0)
    array_h = stripe_img_h.sum(axis=1)


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    img_array_0 = img[:, :, 0]
    img_array_1 = img[:, :, 1]
    img_array_2 = img[:, :, 2]
    stripe_img_w0 = img_array_0[h_stripe[0]:h_stripe[1], :]
    stripe_img_h0 = img_array_0[:, w_stripe[0]:w_stripe[1]]
    stripe_img_w1 = img_array_1[h_stripe[0]:h_stripe[1], :]
    stripe_img_h1 = img_array_1[:, w_stripe[0]:w_stripe[1]]
    stripe_img_w2 = img_array_2[h_stripe[0]:h_stripe[1], :]
    stripe_img_h2 = img_array_2[:, w_stripe[0]:w_stripe[1]]

    array_w0 = stripe_img_w0.sum(axis=0)
    array_w1 = stripe_img_w1.sum(axis=0)
    array_w2 = stripe_img_w2.sum(axis=0)
    array_h0 = stripe_img_h0.sum(axis=1)
    array_h1 = stripe_img_h1.sum(axis=1)
    array_h2 = stripe_img_h2.sum(axis=1)

    rez = np.polyfit(range(w), array_w0, 4)
    f = np.poly1d(rez)
    df = np.polyder(f)
    new_sample_w0 = abs(df(range(w)))-abs(df(0)/2)
    # plt.plot(new_sample_w0)

    rez = np.polyfit(range(w), array_w1, 4)
    f = np.poly1d(rez)
    df = np.polyder(f)
    new_sample_w1 = abs(df(range(w)))-abs(df(0)/2)
    # plt.plot(new_sample_w1)

    rez = np.polyfit(range(w), array_w2, 4)
    f = np.poly1d(rez)
    df = np.polyder(f)
    new_sample_w2 = abs(df(range(w)))-abs(df(0)/2)
    # plt.plot(new_sample_w2)

    rez = np.polyfit(range(h), array_h0, 4)
    f = np.poly1d(rez)
    df = np.polyder(f)
    new_sample_h0 = abs(df(range(h))) - abs(df(0) / 2)
    # plt.plot(new_sample_w2)

    rez = np.polyfit(range(h), array_h1, 4)
    f = np.poly1d(rez)
    df = np.polyder(f)
    new_sample_h1 = abs(df(range(h))) - abs(df(0) / 2)
    # plt.plot(new_sample_w2)

    rez = np.polyfit(range(h), array_h2, 4)
    f = np.poly1d(rez)
    df = np.polyder(f)
    new_sample_h2 = abs(df(range(h))) - abs(df(0) / 2)
    # plt.plot(new_sample_w2)

    new_sample_w = (new_sample_w0+new_sample_w1+new_sample_w2)/3
    new_sample_h = (new_sample_h0 + new_sample_h1 + new_sample_h2) / 3
    # plt.plot((new_sample_w0+new_sample_w1+new_sample_w2)/3)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # # Сигнал скачков яркости
    # # d = (array_1 + array_2+array_3+array_4)/4
    # # d = (array_1 + array_2)/2
    # # d = (array_3 + array_4)/2
    #
    # # Вспомогательные параметры для преобразования Фурье
    # sample_size = 2 ** 3  # 32
    #
    # # Быстрое преобразование Фурье
    # # spectrum = np.fft.fft(d)
    # spectrum_h = np.fft.fft(array_h)
    # spectrum_w = np.fft.fft(array_w)
    # # Фильтрация сигнала
    # pos_freq_i = np.arange(1, sample_size // 2, dtype=int)
    # # psd = np.abs(spectrum[pos_freq_i]) ** 2 + np.abs(spectrum[-pos_freq_i]) ** 2
    #
    # psd_h = np.abs(spectrum_h[pos_freq_i]) ** 2 + np.abs(spectrum_h[-pos_freq_i]) ** 2
    # psd_w = np.abs(spectrum_w[pos_freq_i]) ** 2 + np.abs(spectrum_w[-pos_freq_i]) ** 2
    #
    # # filtered = pos_freq_i[psd > 1e3]
    # filtered_h = pos_freq_i[psd_h > 1e3]
    # filtered_w = pos_freq_i[psd_w > 1e3]
    #
    # # Новый спектр
    # # new_spec = np.zeros_like(spectrum)
    # # new_spec[filtered] = spectrum[filtered]
    # # new_spec[-filtered] = spectrum[-filtered]
    #
    # new_spec_h = np.zeros_like(spectrum_h)
    # new_spec_h[filtered_h] = spectrum_h[filtered_h]
    # new_spec_h[-filtered_h] = spectrum_h[-filtered_h]
    #
    # new_spec_w = np.zeros_like(spectrum_w)
    # new_spec_w[filtered_w] = spectrum_h[filtered_w]
    # new_spec_w[-filtered_w] = spectrum_h[-filtered_w]
    #
    # # Обратное преобразование Фурье
    # # new_sample = np.real(np.fft.ifft(new_spec))
    # new_sample_h = np.real(np.fft.ifft(new_spec_h))
    # new_sample_w = np.real(np.fft.ifft(new_spec_w))

    # ===============================================
    # rez = np.polyfit(range(w), array_w, 4)
    # f = np.poly1d(rez)
    # df = np.polyder(f)
    # new_sample_w = abs(df(range(w)))-abs(df(0))*0.6180

    # rez = np.polyfit(range(h), array_h, 4)
    # f = np.poly1d(rez)
    # df = np.polyder(f)
    # new_sample_h = abs(df(range(h)))-abs(df(0))*0.6180

    # plt.plot(new_sample_h)
    # plt.plot(new_sample_w)
    # ===============================================
    # Если есть смена знака, то точку считаем внутренней границей рамки проверяем до 10% слева и справа
    top_border = 0
    bottom_border = -1
    left_border = 0
    right_border = -1
    r = round(len(new_sample_h)*frame_prc)
    rz = 8
    for i in range(r):
        if new_sample_h[i] * new_sample_h[i + 1] < 0:
            if i > rz:
                top_border = i-rz
            break
    for i in range(r):
        if new_sample_h[-i - 1] * new_sample_h[-i - 2] < 0:
            if i > rz:
                bottom_border = rz-i
            break

    r = round(len(new_sample_w) * frame_prc)
    rz = 5
    for i in range(r):
        if new_sample_w[i] * new_sample_w[i + 1] < 0:
            if i > rz:
                left_border = i-rz
            break
    for i in range(r):
        if new_sample_w[-i - 1] * new_sample_w[-i - 2] < 0:
            if i > rz:
                right_border = -i+rz
            break
    # plt.imshow(img)
    # plt.plot(array_h)
    # plt.plot(new_sample_h)
    # plt.plot(new_spec_h[10: -10])
    plt.show()
    # Обрезаем рамочку
    return (img[top_border:h+bottom_border, left_border:w+right_border])


# def adjust_gamma(image, gamma=1.0):
# 	# build a lookup table mapping the pixel values [0, 255] to
# 	# their adjusted gamma values
# 	invGamma = 1.0 / gamma
# 	table = np.array([((i / 255.0) ** invGamma) * 255
# 		for i in np.arange(0, 256)]).astype("uint8")
#
# 	# apply gamma correction using the lookup table
# 	return cv.LUT(image, table)


def image_processing(img):
    # img_without_frame1 = remove_frame(img, 0.06)
    # img_without_frame2 = remove_frame(img_without_frame1, 0.14)
    # img_after = get_without_back3(img_without_frame2)

    img_after = remove_frame(img, 0.06)
    img_after = remove_frame(img_after, 0.14)
    img_after = get_without_back3(img_after)
    img_after = get_without_back4(img_after)
    return img_after
