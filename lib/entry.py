import sys
import shutil
import argparse
import os

import numpy as np
import cv2 as cv


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