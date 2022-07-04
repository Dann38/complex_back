import fnmatch
import os

import cv2
from img_processing import image_processing
from lib.text_evaluation import checking_for_improvement
from lib.img_info import get_text_from_img, print_statistic, create_statistic_file

from lib.entry import get_input_output_folder, read_img, get_coefficients_reg_model


def main():
    input_folder, output_folder = get_input_output_folder()

    statistical_information = {
        "n": 0,
        "total_similarity_before": 0,
        "total_similarity_after": 0,
        "total_levenshtein_before": 0,
        "total_levenshtein_after": 0,
        "about_images": []
    }
    coefficients_reg_model = get_coefficients_reg_model("coefficients_reg_model.dat")
    coefficients_reg_model = None
    statistics_print = True

    for root, directories, files in os.walk(input_folder):
        for file_name in fnmatch.filter(files, "*.jpg"):
            path = os.path.join(root, file_name)
            name = os.path.splitext(os.path.split(path)[1])[0]

            save_img_path = os.path.join(output_folder, name + ".jpeg")
            save_text_path = os.path.join(output_folder, name + ".txt")
            save_origin_text_path = os.path.join(output_folder, name + "-origin" + ".txt")

            img_before = read_img(path)
            img_after = image_processing(img_before,
                                         statistical_information=statistical_information,
                                         data_reg_model=coefficients_reg_model)

            text_before = get_text_from_img(img_before)
            text_after = get_text_from_img(img_after)

            gs_name = os.path.join(root, name + ".txt")

            try:
                checking_for_improvement(text_after, text_before, gs_name, statistical_information, name)
            except FileNotFoundError:
                print("OK")

            with open(save_text_path, 'w', encoding="UTF-8") as f:
                f.write(text_after)

            with open(save_origin_text_path, 'w', encoding="UTF-8") as f:
                f.write(text_before)

            cv2.imwrite(save_img_path, img_after)

    if statistics_print:
        print_statistic(statistical_information)
        create_statistic_file(output_folder, statistical_information)


if __name__ == '__main__':
    main()
