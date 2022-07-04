import os.path
import random
import re

import cv2 as cv
import numpy as np
import pytesseract

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


def info(img):
    """
    Возвращает массив с информацией о изображение
    :param img: исходное изображение;
    :return: [число пикселей, яркость]
    """
    height, width = img.shape[:2]
    luminance = img.sum() / (width * height * 3 * 255)
    count_px = height * width
    return [count_px, luminance]


def create_statistic_file(folder, statistical_information):
    array = statistical_information["about_images"]
    with open(os.path.join(folder, "data.npy"), 'wb') as f:
        np.save(f, np.array(array))

