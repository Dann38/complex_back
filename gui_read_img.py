from turtle import Canvas

from my_lib import *
from tkinter import *
from PIL import Image, ImageTk
import cv2 as cv
import os
from sys import argv
from pathlib import Path


def main(name_file):
    dir_ = os.getcwd()
    f = os.path.join(dir_, name_file)
    img = read_img(f)

    height, width = img.shape[:2]
    WIDTH = 900
    HEIGHT = WIDTH * height // width

    only_text = get_without_back3(img)

    rez1 = get_text_from_img(img)
    rez2 = get_text_from_img(only_text)

    # Графический интерфейс =====================================================
    root = Tk()
    root.title("GUI на Python")
    root.geometry("1600x800")

    # Картинка до ===============================================================
    canva1 = Canvas(root, width=WIDTH / 2, height=HEIGHT / 2, bg="gray")
    canva1.grid(row=1, column=1)

    canvawidth = int(canva1.winfo_reqwidth())
    canvaheight = int(canva1.winfo_reqheight())
    sp = img.shape
    cvheight = sp[0]  # height(rows) of image
    cvwidth = sp[1]  # width(colums) of image
    img = cv.resize(img, (canvawidth, canvaheight), interpolation=cv.INTER_AREA)

    imgCV2 = cv.cvtColor(img, cv.COLOR_BGR2RGBA)  # Convert color from BGR to RGBA
    current_image = Image.fromarray(imgCV2)  # Convert image into Image object
    imgTK1 = ImageTk.PhotoImage(image=current_image)  # Convert image object to imageTK object
    canva1.create_image(0, 0, anchor=NW, image=imgTK1)

    label1 = Label(text=rez1, justify=LEFT)
    label1.grid(row=1, column=2)


    # Картинка после ============================================================
    canva2 = Canvas(root, width=WIDTH / 2, height=HEIGHT / 2, bg="gray")
    canva2.grid(row=1, column=3)

    canvawidth = int(canva2.winfo_reqwidth())
    canvaheight = int(canva2.winfo_reqheight())
    sp = only_text.shape
    cvheight = sp[0]  # height(rows) of image
    cvwidth = sp[1]  # width(colums) of image
    only_text = cv.resize(only_text, (canvawidth, canvaheight), interpolation=cv.INTER_AREA)

    imgCV2 = cv.cvtColor(only_text, cv.COLOR_BGR2RGBA)  # Convert color from BGR to RGBA
    current_image = Image.fromarray(imgCV2)  # Convert image into Image object
    imgTK2 = ImageTk.PhotoImage(image=current_image)  # Convert image object to imageTK object
    canva2.create_image(0, 0, anchor=NW, image=imgTK2)

    label2 = Label(text=rez2, justify=LEFT)
    label2.grid(row=1, column=4)

    # Конец ======================================================================
    root.mainloop()


if __name__ == '__main__':

    if len(argv) > 1:
        if argv[1] == "-h" or argv[1] == "--help":
            print("Для запуска укажите путь до файла")
        elif not Path(argv[1]).is_file():
            print("Нет такого файла")
        else:
            main(argv[1])
    else:
        print("Для помощи вызовите команду -h или --help")
