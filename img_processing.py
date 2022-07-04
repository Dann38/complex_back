from lib.entry import get_coefficients_reg_model
from lib.luminance import balance_brightness
from lib.blur import *
from lib.binary import *
from lib.frame import *
from lib.img_info import *


def image_processing(img, data_reg_model=None, statistical_information=None):
    """
    :param img:
    :return: исправленное изображение
    """
    IMPROVEMENT_MORE_THAN = 0.05
    NORMA_PX_FORM_BINARY = 0.5e+6

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
    # img_without_frame1 = remove_frame_by_sharp_jump(img, 0.06)
    # img_without_frame2 = remove_frame_by_sharp_jump(img_without_frame1, 0.14)
    # img_after = convolution_method(img_without_frame2)
    # # =========================================================================

    # # ШЕСТАЯ ЭВРИСТИКА =========================================================
    # img_after = golden_ratio_back(img)
    # # ==========================================================================
    # # СЕДЬМАЯ ЭВРИСТИКА ========================================================
    # img_after = puzzles(img)
    # # ==========================================================================
    data_img = info(img)

    img_after = remove_white_back(img, 0.1)
    img_after = remove_frame_by_outlier(img_after, 0.15)

    img_after = balance_brightness(img_after)
    img_after = convolution_method(img_after)
    if data_img[0] > NORMA_PX_FORM_BINARY:
        img_after = puzzles(img_after)

    data_img_after = info(img_after)

    if data_reg_model is None:
        mark = 1  # Число заранее большее, чем IMPROVEMENT_MORE_THAN
    else:
        # mark = c0 + c1 x1 + c2 x2 + ... cn xn
        # ci - коэффициент data_reg_model
        # xi - либо элемент из data_img либо из data_img_after
        mark = (np.array(data_reg_model) * [1, *data_img, *data_img_after]).sum()

    if mark < IMPROVEMENT_MORE_THAN:
        img_after = img

    if statistical_information is not None:
        statistical_information["about_images"].append([*data_img, *data_img_after])

    return img_after
