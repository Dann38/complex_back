import random
import re

import cv2 as cv
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
    height, width = img.shape[:2]
    dh = height // 10
    dw = width // 10
    run = True
    k = 0

    size_litter = 0
    while run and k < 10:
        h_point = 2*dh + round(random.random()*(height-4*dh))
        w_point = 2*dw + round(random.random()*(width-4*dw))
        img2 = img[h_point - dh:h_point+dh, w_point - dw:w_point+dw, :]
        boxes = pytesseract.image_to_boxes(img2, config=CONFIG_TESSERACT)

        boxes = boxes.splitlines()
        n = len(boxes)
        str_ = ''.join([item[0] for item in boxes])
        match = re.search(r'([А-Яа-я]){3}', str_)

        if match is None:
            k += 1
            continue
        else:
            run = False
        boxes = [boxes[i] for i in range(*match.span())]

        for item in boxes:
            item = item.split()
            size_litter = size_litter + int(item[4]) - int(item[2])
        size_litter = size_litter/3

    info_ = f"""
    height: {height} \t width: {width} \t({height * width})
    yar: {img.sum() / (width * height * 3 * 255):5.2f}
    mean: {img.mean():5.2f}
    font: {round(size_litter*0.5)}-{round(size_litter*1.5)} px
"""
    return info_, [height * width, img.sum() / (width * height * 3 * 255), img.mean(), size_litter]