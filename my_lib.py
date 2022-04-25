import random

import cv2 as cv
import pytesseract
import difflib
import numpy as np


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
    # img2 = cv.medianBlur(img2, 3)
    kernel = np.array([[0, 0, 0],
                       [-1, 7, 1],
                       [0, 0, 0]])

    kernel = 1/5*kernel
    img2 = cv.filter2D(img, -1, kernel)
    return img2

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
