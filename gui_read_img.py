from tkinter import *
from PIL import Image, ImageTk
import cv2 as cv
from img_processing import image_processing
from lib.entry import get_img
from lib.img_info import get_info_from_img
from lib.binarization import binarize

class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("GUI на Python")
        self.root.geometry("1600x800")
        img = get_img()
        height, width = img.shape[:2]
        self.WIDTH = 900
        self.HEIGHT = self.WIDTH * height // width

        only_text = image_processing(img)

        rez2, only_text = get_info_from_img(only_text)
        rez1, img = get_info_from_img(img)
        # img_rez = binarize(img)
        # rez1, img = get_info_from_img(img_rez)
        self.imgs = [img, only_text]

        self.img_and_text_gui_block(rez1, 0)
        self.img_and_text_gui_block(rez2, 1)

        self.root.mainloop()

    def img_and_text_gui_block(self, text_img, index_img):
        canva = Canvas(self.root, width=self.WIDTH // 2, height=self.HEIGHT // 2, bg="gray")
        canva.pack(side=LEFT)

        canvawidth = int(canva.winfo_reqwidth())
        canvaheight = int(canva.winfo_reqheight())

        self.imgs[index_img] = self.array_to_image_tk(self.imgs[index_img], canvawidth, canvaheight)
        canva.create_image(0, 0, image=self.imgs[index_img], anchor='nw')

        text = Text(self.root, width=40, height=30)
        text.pack(side=LEFT)

        scroll = Scrollbar(command=text.yview)
        scroll.pack(side=LEFT, fill=Y)

        text.config(yscrollcommand=scroll.set)
        text.insert('1.0', text_img)

    @staticmethod
    def array_to_image_tk(img, width, height):
        img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)
        img_cv = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
        current_image = Image.fromarray(img_cv)
        return ImageTk.PhotoImage(current_image)


if __name__ == '__main__':
    app = App()
