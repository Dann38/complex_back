import fnmatch
import getopt
import os
import shutil
import sys
from sys import argv
from pathlib import Path

import cv2

from my_lib import get_text_from_img, get_without_back, similarity, read_img


def main(argv):
    input_folder = ''
    output_folder = ''

    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["input_folder=", "output_folder="])
    except getopt.GetoptError:
        print('script.py -i <input folder> -o <output folder>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('script.py -i <input folder> -o <output folder>')
            sys.exit()
        elif opt in ("-i", "--i"):
            input_folder = arg
        elif opt in ("-o", "--o"):
            output_folder = os.path.join(arg, 'images')

    is_dir = os.path.isdir(input_folder)

    if not is_dir:
        raise Exception(input_folder, "- It is not a folder")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder)

    n = 0
    sum = 0
    for root, directories, files in os.walk(input_folder):
        for file_name in fnmatch.filter(files, "*.jpg"):
            path = os.path.join(root, file_name)
            name = os.path.splitext(os.path.split(path)[1])[0]
            save_path = os.path.join(output_folder, name + ".jpeg")
            img_before = cv2.imread(path)
            text_before = get_text_from_img(img_before)
            img_after = get_without_back(img_before, )
            cv2.imwrite(save_path, img_after)
            text_after = get_text_from_img(img_after)
            rez = similarity(text_before, text_after)
            print(f"Похожесть до и после: {100 * rez :5.2f} %")
            sum += rez
            n += 1

    print("ИТОГ МЕТОДА =======================")
    print(f"{100*sum/n:5.2f} %")


if __name__ == '__main__':
    main(argv[1:])
