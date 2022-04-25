import fnmatch
import getopt
import os
import shutil
import sys
from sys import argv
from Levenshtein import distance
import cv2

from my_lib import get_text_from_img, get_without_back, similarity, read_img, get_without_back3


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
    total_similarity_before = 0
    total_similarity_after = 0
    total_levenshtein_before = 0
    total_levenshtein_after = 0
    total_size = 0
    statistics_print = False
    bad_image = "None"
    bad_similarity = 2
    delta_similary = 0

    for root, directories, files in os.walk(input_folder):
        for file_name in fnmatch.filter(files, "*.jpg"):
            path = os.path.join(root, file_name)
            name = os.path.splitext(os.path.split(path)[1])[0]

            save_img_path = os.path.join(output_folder, name + ".jpeg")
            save_text_path = os.path.join(output_folder, name + ".txt")
            save_origin_text_path = os.path.join(output_folder, name + "-origin" + ".txt")

            # img_before = cv2.imread(path)
            img_before = read_img(path)
            img_after = get_without_back3(img_before)

            text_before = get_text_from_img(img_before)
            text_after = get_text_from_img(img_after)

            gs_name = os.path.join(root, name + ".txt")
            try:
                with open(gs_name, 'r', encoding="UTF-8") as f:
                    text_gs = f.read()
                    similarity_before = similarity(text_gs, text_before)
                    similarity_after = similarity(text_gs, text_after)
                    total_similarity_before += similarity_before
                    total_similarity_after += similarity_after

                    size = len(text_gs)
                    total_size += size
                    levenshtein_before = distance(text_gs, text_before)
                    levenshtein_after = distance(text_gs, text_after)
                    total_levenshtein_before += levenshtein_before
                    total_levenshtein_after += levenshtein_after
                    print()
                    print(
                        f"{name}:\tSimilarity: \t"
                        f" Before: {100 * similarity_before :5.2f}%\t"
                        f" After: {100 * similarity_after :5.2f}%")
                    print(
                        f"{name}:\tLevenshtein:\t"
                        f" Before: {levenshtein_before / size :5.2f}\t"
                        f" After: {levenshtein_after / size:5.2f}")
                    print("=======================")
                    delta_similary = similarity_after-similarity_before
                    if bad_similarity > delta_similary:
                        bad_image = name
                        bad_similarity = delta_similary
                    statistics_print = True
                    n += 1
            except FileNotFoundError:
                print("OK")

            with open(save_text_path, 'w', encoding="UTF-8") as f:
                f.write(text_after)

            with open(save_origin_text_path, 'w', encoding="UTF-8") as f:
                f.write(text_before)

            cv2.imwrite(save_img_path, img_after)

    if statistics_print:
        print("Total:")
        print(f"Before:\t {100 * total_similarity_before / n:5.2f} %")
        print(f"After:\t {100 * total_similarity_after / n:5.2f} %")
        print("Total Levenshtein =======================")
        print(f"Before:\t {total_levenshtein_before / total_size:5.2f}")
        print(f"After:\t {total_levenshtein_after / total_size:5.2f}")
        print(f"Bad image:\t  {bad_image}\t (delta:{delta_similary})")


if __name__ == '__main__':
    main(argv[1:])
