from my_lib import get_img, get_info_from_img, golden_ratio_back
from tkinter import *
from PIL import Image, ImageTk
import cv2 as cv


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("GUI на Python")
        self.root.geometry("1600x800")
        img = get_img()
        height, width = img.shape[:2]
        self.WIDTH = 900
        self.HEIGHT = self.WIDTH * height // width

        rez1, img = get_info_from_img(img)
        self.imgs = [img]

        self.img_and_text_gui_block(rez1, 0)
        img2 = golden_ratio_back(img[height//2-100:height//2+100, width//2-100:width//2+100, :])
        cv.imshow("", img2)
        info = f"""
height: {height} \t width: {width} \t({height*width})
yar: {img.sum() / (width * height * 3 * 255):5.2f}
mean: {img.mean():5.2f}
        """
        text = Text(self.root, width=40, height=30)
        text.pack(side=LEFT)
        text.insert('1.0', info)
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