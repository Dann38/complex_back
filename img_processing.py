from lib.luminance import balance_brightness
from lib.blur import *
from lib.binary import *
from lib.frame import *
from lib.img_info import *


def image_processing(img):
    """
    :param img:
    :return: исправленное изображение
    """
    # # ПЕРВАЯ ЭВРИСТИКА ========================================================
    # img_after = subtracting_white_spots(img)
    # # =========================================================================

    # # ВТОРАЯ ЭВРИСТИКА ========================================================
    # img_after = sharpening(img)
    # # =========================================================================

    # # ТРЕТЬЯ ЭВРИСТИКА ========================================================
    # img_after = balance_brightness(img)
    # # =========================================================================

    # # ЧЕТВЕРТАЯ ЭВРИСТИКА =====================================================
    # img_after = convolution_method(img)
    # # =========================================================================

    # # ПЯТАЯ ЭВРИСТИКА =========================================================
    # img_without_frame1 = remove_frame(img, 0.06)
    # img_without_frame2 = remove_frame(img_without_frame1, 0.14)
    # img_after = convolution_method(img_without_frame2)
    # # =========================================================================

    # # ШЕСТАЯ ЭВРИСТИКА =========================================================
    # img_after = golden_ratio_back(img)
    # # ==========================================================================
    # # СЕДЬМАЯ ЭВРИСТИКА ========================================================
    # img_after = puzzles(img)
    # # ==========================================================================

    info_, data = info(img)
    mark = 0.773 - 0.94 * data[1]
    if mark < 0.05:
        img_after = img
    else:
        img_after = remove_white_back(img, 0.1)
        img_after = remove_frame_by_outlier(img_after, 0.15)
        img_after = balance_brightness(img_after)
        img_after = convolution_method(img_after)
        img_after = puzzles(img_after)

    return img_after