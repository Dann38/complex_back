import random
import sys

import cv2 as cv
import pytesseract
import difflib
import numpy as np
import matplotlib.pyplot as plt

import shutil
import argparse
import os

from Levenshtein import distance
CONFIG_TESSERACT = '-l rus+eng --oem 3 --psm 3'
"""
-l - язык
    0. rus
    1. rus+eng
--oem - какой движок использовать
    0. Устаревший движок
    1. Нейронная сеть LSTM
    2. Устаревший движок и LSTM
    3. По умолчанию

--psm - сегментация страницы
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
"""


def read_img(name_file):
    """
    Позволяет открывать изображения с разными именами,
    в случае когда имя содержит киррилические символы,
    возникают проблемы чтения функции cv2.imread
    :param name_file: имя файла;
    :return: изображение как объект
    """
    with open(name_file, "rb") as f:
        chunk = f.read()
    chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
    img = cv.imdecode(chunk_arr, cv.IMREAD_COLOR)
    return img


def get_text_from_img(img, custom_config=CONFIG_TESSERACT):
    text = pytesseract.image_to_string(img, config=custom_config)
    return text


def get_img_selected_text(img, custom_config=CONFIG_TESSERACT):
    shape = img.shape
    h, w = shape[0], shape[1]

    boxes = pytesseract.image_to_boxes(img, config=custom_config)

    for b in boxes.splitlines():
        b = b.split()
        cv.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 1)

    return img


def get_info_from_img(img):
    text = get_text_from_img(img)
    img = get_img_selected_text(img)

    return text, img


def similarity(s1, s2):
    normalized1 = s1.lower()
    normalized2 = s2.lower()
    matcher = difflib.SequenceMatcher(None, normalized1, normalized2)
    return matcher.ratio()


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


def convolution_method(img):
    """
    Свертка. Значение пикселя изображения зависит также от соседних пикселей.
    :param img: Исходное изображение;
    :return: После свертки.
    """
    # kernel = np.array([[1, 0, -1],
    #                    [2, 6, -2],
    #                    [1, 0, -1]])
    kernel = np.array([[1, 0, -1],
                       [1, 6, 1],
                       [1, 0, -1]])

    kernel = 1/(kernel.sum())*kernel

    img2 = cv.filter2D(img, -1, kernel)

    return img2


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
    img = cv.filter2D(img, -1, 1/yar)
    return img


def get_without_back4(img):
    x = img.ravel()
    up = x.mean()/1.618
    _, img2 = cv.threshold(img, up, 255, 0)

    return img + img2


def approximation_polynomial(array):
    n = len(array)
    rez = np.polyfit(range(n), array, 4)
    f = np.poly1d(rez)
    df = np.polyder(f)
    new_sample = abs(df(range(n))) - abs(df(0) / 2)
    return new_sample


def remove_frame(img, frame_prc=0.1):
    sh = img.shape
    h, w = sh[0], sh[1]

    array_w0 = img[::10, :, 0].sum(axis=0)
    array_h0 = img[:, ::10, 0].sum(axis=1)
    array_w1 = img[::10, :, 1].sum(axis=0)
    array_h1 = img[:, ::10, 1].sum(axis=1)
    array_w2 = img[::10, :, 2].sum(axis=0)
    array_h2 = img[:, ::10, 2].sum(axis=1)

    new_sample_w0 = approximation_polynomial(array_w0)
    new_sample_w1 = approximation_polynomial(array_w1)
    new_sample_w2 = approximation_polynomial(array_w2)
    new_sample_h0 = approximation_polynomial(array_h0)
    new_sample_h1 = approximation_polynomial(array_h1)
    new_sample_h2 = approximation_polynomial(array_h2)

    new_sample_w = (new_sample_w0+new_sample_w1+new_sample_w2)/3
    new_sample_h = (new_sample_h0 + new_sample_h1 + new_sample_h2) / 3

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

    plt.show()
    # Обрезаем рамочку
    return (img[top_border:h+bottom_border, left_border:w+right_border])


def remove_frame2(img, proc):
    sh = img.shape
    h, w = sh[0], sh[1]
    img_array = img.sum(axis=2) / (255 * 3)

    delta = round(w*proc)
    array_h = np.zeros(w)
    for i in range(delta, h-delta, 10):
        array_h += img_array[i, :]

    delta = round(h * proc)
    array_w = np.zeros(h)
    for i in range(delta, w-delta, 10):
        array_w += img_array[:, i]

    hampel_h = hampel(array_h)
    hampel_w = hampel(array_w)

    left_border = 0
    right_border = -1
    for i in range(h):
        if i < w*proc and hampel_h[i]:
            left_border = i
    for i in range(1, h):
        if i < w*proc and hampel_h[-i]:
            right_border = -i

    top_border = 0
    bottom_border = -1
    for i in range(w):
        if i < h*proc and hampel_w[i]:
            top_border = i
    for i in range(1, w):
        if i < h*proc and hampel_w[-i]:
            bottom_border = -i

    return img[top_border+h//100:bottom_border-h//100, left_border+w//100:right_border-w//100, :]


def hampel(vals_orig):
    vals = vals_orig
    difference = np.abs(vals.mean()-vals)
    median_abs_deviation = difference.mean()
    threshold = 3 * median_abs_deviation
    outlier_idx = difference > threshold
    return outlier_idx


def image_processing(img):
    """
    :param img:
    :return: исправленное изображение
    """
    # # ПЕРВАЯ ЭВРИСТИКА ========================================================
    # img_after = subtracting_white_spots(img)
    # # =========================================================================

    # # ВТОРАЯ ЭВРИСТИКА ========================================================
    # img_after = sharpening(img)
    # # =========================================================================

    # # ТРЕТЬЯ ЭВРИСТИКА ========================================================
    # img_after = balance_brightness(img)
    # # =========================================================================

    # # ЧЕТВЕРТАЯ ЭВРИСТИКА =====================================================
    # img_after = convolution_method(img)
    # # =========================================================================

    # # ПЯТАЯ ЭВРИСТИКА =========================================================
    img_without_frame1 = remove_frame(img, 0.06)
    img_without_frame2 = remove_frame(img_without_frame1, 0.14)
    img_after = convolution_method(img_without_frame2)
    # # =========================================================================

    # img_after = remove_frame2(img, 0.14)
    # img_after = balance_brightness(img_after)
    # img_after = convolution_method(img_after)
    # img_after = get_without_back4(img_after)

    return img_after


def reading_text(img):
    img_after = image_processing(img)
    text_before, img_before = get_info_from_img(img)
    text_after, img_after = get_info_from_img(img_after)


def get_input_output_folder():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="Папка с изображениями для обработки")
    parser.add_argument('-o', '--output', help="Папка для записи результатов")
    parser.add_argument('-ow', '--overwrite', action='store_true', default=False,
                        help="Указывается в случае перезаписи результатов")

    namespace = parser.parse_args()
    input_dir = namespace.input
    output_dir = namespace.output

    is_dir = os.path.isdir(input_dir)
    if not is_dir:
        print(input_dir, "- Не удалось найти папку")
        sys.exit(1)
    if os.path.exists(output_dir) and namespace.overwrite:
        shutil.rmtree(output_dir)
    elif os.path.exists(output_dir):
        print(output_dir, "- Эта папка уже существует. Для перезаписи добавьте -ow или --overwrite")
        sys.exit(1)
    os.makedirs(output_dir)

    return input_dir, output_dir


def checking_for_improvement(text_after, text_before, gs_name, st_info, name_file="IMAGE"):
    with open(gs_name, 'r', encoding="UTF-8") as f:
        wt_1 = 1 / (st_info['n'] + 1)
        wt_n = st_info['n'] * wt_1

        text_gs = f.read()
        similarity_before = similarity(text_gs, text_before)
        similarity_after = similarity(text_gs, text_after)

        st_info["total_similarity_before"] = wt_n * st_info["total_similarity_before"] + wt_1 * similarity_before
        st_info["total_similarity_after"] = wt_n * st_info["total_similarity_after"] + wt_1 * similarity_after

        size = len(text_gs)
        levenshtein_before = distance(text_gs, text_before) / size
        levenshtein_after = distance(text_gs, text_after) / size

        st_info["total_levenshtein_before"] = wt_n * st_info["total_levenshtein_before"] + wt_1 * levenshtein_before
        st_info["total_levenshtein_after"] = wt_n * st_info["total_levenshtein_after"] + wt_1 * levenshtein_after

        print()
        print(
            f"{name_file}:\tSimilarity: \t"
            f" Before: {100 * similarity_before :5.2f}%\t"
            f" After: {100 * similarity_after :5.2f}%")
        print(
            f"{name_file}:\tLevenshtein:\t"
            f" Before: {levenshtein_before:5.2f}\t"
            f" After: {levenshtein_after:5.2f}")
        print("=======================")

        st_info["n"] += 1


def print_statistic(statistical_information):
    lev_before = statistical_information["total_levenshtein_before"]
    lev_after = statistical_information["total_levenshtein_after"]
    lev_improvement_percent = (lev_before - lev_after) * 100
    print("Total Similarity ========================")
    print(f"Before:\t {100 * statistical_information['total_similarity_before']:5.2f} %")
    print(f"After:\t {100 * statistical_information['total_similarity_after']:5.2f} %")
    print("Total Levenshtein =======================")
    print(f"Before:\t {lev_before:5.2f}")
    print(f"After:\t {lev_after:5.2f}")
    print(f"Improvement  Levenshtein: {lev_improvement_percent:5.2f}%")
