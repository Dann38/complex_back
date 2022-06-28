import math
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
CONFIG_TESSERACT = '-l rus --oem 3 --psm 3'
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
    # kernel = np.array([[-1, -1, -1],
    #                    [-1, 8, -1],
    #                    [-1, -1, -1]])
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


def sharp_jump(array):
    """
    Преобразование массива таким образом, чтобы пересечение значения 0
    совпадало с резким изменением спектра.
    :param array: Исходный массив;
    :return: Новый массив.
    """
    n = len(array)
    rez = np.polyfit(range(n), array, 4)
    f = np.poly1d(rez)
    df = np.polyder(f)
    new_sample = abs(df(range(n))) - abs(df(0) / 2)
    return new_sample


def where_sharp_jump(array, frame_prc):
    """
    Находит точку смены знака ближайшую к центру, но в пределах выбранного процента.
    :param array: Массив обработанный функцией sharp_jump;
    :param frame_prc: Часть на концах, где ожидается скачок (от 0.0 до 1.0);
    :return: Левый и правый индекс в массиве.
    """
    top_border = 0
    bottom_border = -1
    r = round(len(array) * frame_prc)
    rz = 5
    for i in range(r):
        if array[i] * array[i + 1] < 0:
            if i > rz:
                top_border = i - rz
            break
    for i in range(r):
        if array[-i - 1] * array[-i - 2] < 0:
            if i > rz:
                bottom_border = rz - i
            break
    return top_border, bottom_border


def remove_frame_by_sharp_jump(img, frame_prc=0.1):
    """
    Удаляет рамку в районе выбранного процента. Функция выделяет 3 канала в изображение
    находит резкие скачки в них и по ним выделяет рамку.
    :param img: Исходное изображение;
    :param frame_prc: Часть изображения, которую может занимать рамка с каждой стороны(0.0 до 1.0).
    :return: Обрезанное изображение.
    """
    sh = img.shape
    h, w = sh[0], sh[1]

    # Выделение 3-х каналов
    array_w0 = img[::10, :, 0].sum(axis=0)
    array_h0 = img[:, ::10, 0].sum(axis=1)
    array_w1 = img[::10, :, 1].sum(axis=0)
    array_h1 = img[:, ::10, 1].sum(axis=1)
    array_w2 = img[::10, :, 2].sum(axis=0)
    array_h2 = img[:, ::10, 2].sum(axis=1)

    # Преобразование массивов
    new_sample_w0 = sharp_jump(array_w0)
    new_sample_w1 = sharp_jump(array_w1)
    new_sample_w2 = sharp_jump(array_w2)
    new_sample_h0 = sharp_jump(array_h0)
    new_sample_h1 = sharp_jump(array_h1)
    new_sample_h2 = sharp_jump(array_h2)

    new_sample_w = (new_sample_w0 + new_sample_w1 + new_sample_w2) / 3
    new_sample_h = (new_sample_h0 + new_sample_h1 + new_sample_h2) / 3

    # Поиск границы по скачкам в спектре
    top_border, bottom_border = where_sharp_jump(new_sample_w, frame_prc)
    left_border, right_border = where_sharp_jump(new_sample_h, frame_prc)

    return img[top_border:h+bottom_border, left_border:w+right_border]


def hampel(vals_orig):
    """
    Фильтр Хэмпеля. Обнаружение статистических выбросов.
    :param vals_orig: Исходный массив;
    :return: Логический массив, где true - есть выброс, false - нет.
    """
    vals = vals_orig
    difference = np.abs(vals.mean()-vals)
    median_abs_deviation = difference.mean()
    threshold = 3 * median_abs_deviation
    outlier_idx = difference > threshold
    return outlier_idx


def max_len(delta1, delta2):
    """
    Из массива отрезков выбирает наибольшую длину отрезка (суммируются только соседние отрезки).
    :param delta1: Первый массив расстояний;
    :param delta2: Второй массив расстояний;
    :return: Максимальную длину плюс то что в начале.
    """

    # Маленькие отрезки (длина не больше 2) прибавляются к соседним
    delta1_new = [0]
    delta2_new = [0]
    for i in delta1:
        if i <= 1:
            delta1_new[-1] += i
        else:
            delta1_new.append(i)

    for i in delta2:
        if i <= 2:
            delta2_new[-1] += i
        else:
            delta2_new.append(i)

    # Функция для многомерной оптимизации
    # Максимизируется длина отрезков состоящего из мелких
    # Минимизируется их отличие
    def f_sum_delta(x):
        start1, stop1, start2, stop2 = x
        if stop1 - start1 <= 0 or stop2 - start2 <= 0:
            return - math.inf

        sum_delta1 = sum(delta1_new[start1:stop1])
        sum_delta2 = sum(delta1_new[start2:stop2])
        return sum_delta1 + sum_delta2 - 0.5*abs(sum_delta1 - sum_delta2)

    # Поиск лучшего варианта полным перебором
    n1 = len(delta1_new)
    n2 = len(delta2_new)

    max_rez = -1
    max_x = [0, 0, 0, 0]
    for start_1 in range(n1):
        for d1 in range(n1-start_1):
            for start_2 in range(n2):
                for d2 in range(n2 - start_2):
                    x_new = [start_1, start_1+d1, start_2, start_2+d2]
                    rez_new = f_sum_delta(x_new)
                    if f_sum_delta(x_new) > max_rez:
                        max_rez = rez_new
                        max_x = x_new

    return sum(delta1_new[0:max_x[1]]), sum(delta2_new[0:max_x[3]])


def where_frame_by_outlier(array, proc):
    """
    Определение рамки как точек выброса.
    :param array: Массив характеризующий изображение;
    :param proc: Часть на концах, где ожидается скачок (от 0.0 до 1.0);
    :return: Левый и правый индекс в массиве.
    """
    # Фильтр Хэмпел
    hampel_array = hampel(array)

    n = len(array)
    left_borders = [0]
    right_borders = [-1]

    # Находим точки, где есть выбросы
    for i in range(round(n*proc)):
        if hampel_array[i]:
            left_borders.append(i)

    for i in range(1, round(n*proc)):
        if hampel_array[-i]:
            right_borders.append(-i)

    # Формируем массивы длин отрезков
    delta_left = []
    for i in range(1, len(left_borders)):
        delta_left.append(left_borders[i] - left_borders[i-1])

    delta_right = []
    for i in range(1, len(right_borders)):
        delta_right.append(right_borders[i-1] - right_borders[i])

    # Находим максимальные длины плюс то что до
    left_border, right_border = max_len(delta_left, delta_right)

    return left_border, right_border


def remove_frame_by_outlier(img, proc):
    """
    Удаление рамки через выбросы. По усредненным массивам (полоса по ширине и полоса по высоте)
    определяются границы рамки.
    :param img: Исходное изображение;
    :param proc: Часть изображения, которую может занимать рамка с каждой стороны(0.0 до 1.0).
    :return: Изображение без рамки.
    """
    sh = img.shape
    h, w = sh[0], sh[1]
    img_array = img.sum(axis=2) / (255 * 3)

    delta = round(w*proc)
    array_w = np.zeros(w)
    for i in range(delta, h-delta, 10):
        array_w += img_array[i, :]

    delta = round(h * proc)
    array_h = np.zeros(h)
    for i in range(delta, w-delta, 10):
        array_h += img_array[:, i]

    left_border, right_border = where_frame_by_outlier(array_w, proc)
    top_border, bottom_border = where_frame_by_outlier(array_w, proc)

    return img[top_border:h-bottom_border, left_border:w-right_border, :]


def get_border_white_array(array, proc=0.20):
    """
    Пробегает с краев к центру элементы массива до первого темного элемента (< 245, Яркость от 0 до 255)
    :param array: Массив яркости;
    :param proc: Часть массива, которую достаточно рассмотреть с каждой стороны(0.0 до 1.0).
    :return: Левый и правый индекс в массиве.
    """
    delta = round(len(array) * proc)
    left_border = 0
    right_border = -1

    for i in range(5, delta):
        if array[i] > 245:
            left_border = i
        else:
            break

    for i in range(6, delta):
        if array[-i] > 245:
            right_border = -i
        else:
            break

    return left_border, right_border


def remove_white_back(img, proc=0.20):
    """
    Удаление белой рамки при сканировании.
    :param img: Исходное изображение;
    :param proc: Часть изображения, которую может занимать рамка с каждой стороны(0.0 до 1.0).
    :return: Изображение без белой рамки.
    """
    sh = img.shape
    h, w = sh[0], sh[1]
    img_array = img.sum(axis=2) / 3

    delta = round(w * proc)
    array_w = np.zeros(w)
    k = 0
    for i in range(delta, h - delta, 10):
        array_w += img_array[i, :]
        k += 1
    array_w = 1/k * array_w

    delta = round(h * proc)
    array_h = np.zeros(h)
    k = 0
    for i in range(delta, w - delta, 10):
        array_h += img_array[:, i]
        k += 1
    array_h = 1 / k * array_h

    left_border, right_border = get_border_white_array(array_w, proc)
    top_border, bottom_border = get_border_white_array(array_h, proc)

    return img[top_border:h+bottom_border, left_border:w+right_border, :]


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
    # img_without_frame1 = remove_frame(img, 0.06)
    # img_without_frame2 = remove_frame(img_without_frame1, 0.14)
    # img_after = convolution_method(img_without_frame2)
    # # =========================================================================

    # # ШЕСТАЯ ЭВРИСТИКА =========================================================
    # img_after = golden_ratio_back(img)
    # # ==========================================================================
    img_after = puzzles(golden_ratio_back, img)

    # # ==========================================================================
    # img_after = remove_white_back(img, 0.1)
    # img_after = remove_frame_by_outlier(img_after, 0.15)
    # img_after = balance_brightness(img_after)
    # img_after = convolution_method(img_after)
    # img_after = golden_ratio_back(img_after)

    return img_after


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


def get_img():
    parser = argparse.ArgumentParser()
    parser.add_argument('name_img', help="Путь до изображения")
    namespace = parser.parse_args()
    img = read_img(namespace.name_img)
    return img


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


def puzzles(func, img):
    N = 20
    sh = img.shape
    h, w = sh[0]//N, sh[1]//N
    new_img = img.copy()
    for i in range(N):
        for j in range(N):
            new_img[i*h:(i+1)*h, j*w:(j+1)*w, :] = func(img[i*h:(i+1)*h, j*w:(j+1)*w, :])

    return new_img

