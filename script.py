import os
import cv2 as cv
from my_lib import get_text_from_img, get_without_back, similarity

NAME_DIR = 'IMG'

img_names = os.listdir(NAME_DIR)
dir_ = os.getcwd()

os.chdir(os.path.join(dir_, NAME_DIR))
sum = 0
for img_name in img_names:
    img_before = cv.imread(img_name)
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