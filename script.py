import os
from sys import argv
from pathlib import Path
from my_lib import get_text_from_img, get_without_back, similarity, read_img
NAME_DIR = 'IMG'


def main(name_dir):
    dir_ = os.getcwd()
    dir_ = os.path.join(dir_, name_dir)
    os.chdir(dir_)
    img_names = os.listdir(dir_)

    sum = 0
    for img_name in img_names:
        img_before = read_img(img_name)
        text_before = get_text_from_img(img_before)
        img_after = get_without_back(img_before, )
        text_after = get_text_from_img(img_after)
        rez = similarity(text_before, text_after)
        print(f"Похожесть до и после: {100 * rez :5.2f} %")
        sum += rez
        #print(f"Разница до и после: {100 * (1-rez) :5.2f} %")

    n = len(img_names)
    print("ИТОГ МЕТОДА =======================")
    print(f"{100*sum/n:5.2f} %")


if __name__ == '__main__':
    if len(argv) > 1:
        if argv[1] == "-h" or argv[1] == "--help":
            print("Для запуска создайте директорию IMG или передайте путь до папки")
        elif not Path(argv[1]).is_dir():
            print("Нет такой папки")
        else:
            main(argv[1])
    else:
        print("Для помощи вызовите команду -h или --help")