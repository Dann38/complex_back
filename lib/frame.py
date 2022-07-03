import math
import numpy as np


def sharp_jump(array):
    """
    Преобразование массива таким образом, чтобы пересечение значения 0
    совпадало с резким изменением спектра.
    :param array: Исходный массив;
    :return: Новый массив.
    """
    n = len(array)
    rez = np.polyfit(range(n), array, 4)
    f = np.poly1d(rez)
    df = np.polyder(f)
    new_sample = abs(df(range(n))) - abs(df(0) / 2)
    return new_sample


def where_sharp_jump(array, frame_prc):
    """
    Находит точку смены знака ближайшую к центру, но в пределах выбранного процента.
    :param array: Массив обработанный функцией sharp_jump;
    :param frame_prc: Часть на концах, где ожидается скачок (от 0.0 до 1.0);
    :return: Левый и правый индекс в массиве.
    """
    top_border = 0
    bottom_border = -1
    r = round(len(array) * frame_prc)
    rz = 5
    for i in range(r):
        if array[i] * array[i + 1] < 0:
            if i > rz:
                top_border = i - rz
            break
    for i in range(r):
        if array[-i - 1] * array[-i - 2] < 0:
            if i > rz:
                bottom_border = rz - i
            break
    return top_border, bottom_border


def remove_frame_by_sharp_jump(img, frame_prc=0.1):
    """
    Удаляет рамку в районе выбранного процента. Функция выделяет 3 канала в изображение
    находит резкие скачки в них и по ним выделяет рамку.
    :param img: Исходное изображение;
    :param frame_prc: Часть изображения, которую может занимать рамка с каждой стороны(0.0 до 1.0).
    :return: Обрезанное изображение.
    """
    sh = img.shape
    h, w = sh[0], sh[1]

    # Выделение 3-х каналов
    array_w0 = img[::10, :, 0].sum(axis=0)
    array_h0 = img[:, ::10, 0].sum(axis=1)
    array_w1 = img[::10, :, 1].sum(axis=0)
    array_h1 = img[:, ::10, 1].sum(axis=1)
    array_w2 = img[::10, :, 2].sum(axis=0)
    array_h2 = img[:, ::10, 2].sum(axis=1)

    # Преобразование массивов
    new_sample_w0 = sharp_jump(array_w0)
    new_sample_w1 = sharp_jump(array_w1)
    new_sample_w2 = sharp_jump(array_w2)
    new_sample_h0 = sharp_jump(array_h0)
    new_sample_h1 = sharp_jump(array_h1)
    new_sample_h2 = sharp_jump(array_h2)

    new_sample_w = (new_sample_w0 + new_sample_w1 + new_sample_w2) / 3
    new_sample_h = (new_sample_h0 + new_sample_h1 + new_sample_h2) / 3

    # Поиск границы по скачкам в спектре
    top_border, bottom_border = where_sharp_jump(new_sample_w, frame_prc)
    left_border, right_border = where_sharp_jump(new_sample_h, frame_prc)

    return img[top_border:h+bottom_border, left_border:w+right_border]


def hampel(vals_orig):
    """
    Фильтр Хэмпеля. Обнаружение статистических выбросов.
    :param vals_orig: Исходный массив;
    :return: Логический массив, где true - есть выброс, false - нет.
    """
    vals = vals_orig
    difference = np.abs(vals.mean()-vals)
    median_abs_deviation = difference.mean()
    threshold = 3 * median_abs_deviation
    outlier_idx = difference > threshold
    return outlier_idx


def max_len(delta1, delta2):
    """
    Из массива отрезков выбирает наибольшую длину отрезка (суммируются только соседние отрезки).
    :param delta1: Первый массив расстояний;
    :param delta2: Второй массив расстояний;
    :return: Максимальную длину плюс то что в начале.
    """

    # Маленькие отрезки (длина не больше 2) прибавляются к соседним
    delta1_new = [0]
    delta2_new = [0]
    for i in delta1:
        if i <= 1:
            delta1_new[-1] += i
        else:
            delta1_new.append(i)

    for i in delta2:
        if i <= 2:
            delta2_new[-1] += i
        else:
            delta2_new.append(i)

    # Функция для многомерной оптимизации
    # Максимизируется длина отрезков состоящего из мелких
    # Минимизируется их отличие
    def f_sum_delta(x):
        start1, stop1, start2, stop2 = x
        if stop1 - start1 <= 0 or stop2 - start2 <= 0:
            return - math.inf

        sum_delta1 = sum(delta1_new[start1:stop1])
        sum_delta2 = sum(delta1_new[start2:stop2])
        return sum_delta1 + sum_delta2 - 0.5*abs(sum_delta1 - sum_delta2)

    # Поиск лучшего варианта полным перебором
    n1 = len(delta1_new)
    n2 = len(delta2_new)

    max_rez = -1
    max_x = [0, 0, 0, 0]
    for start_1 in range(n1):
        for d1 in range(n1-start_1):
            for start_2 in range(n2):
                for d2 in range(n2 - start_2):
                    x_new = [start_1, start_1+d1, start_2, start_2+d2]
                    rez_new = f_sum_delta(x_new)
                    if f_sum_delta(x_new) > max_rez:
                        max_rez = rez_new
                        max_x = x_new

    return sum(delta1_new[0:max_x[1]]), sum(delta2_new[0:max_x[3]])


def where_frame_by_outlier(array, proc):
    """
    Определение рамки как точек выброса.
    :param array: Массив характеризующий изображение;
    :param proc: Часть на концах, где ожидается скачок (от 0.0 до 1.0);
    :return: Левый и правый индекс в массиве.
    """
    # Фильтр Хэмпел
    hampel_array = hampel(array)

    n = len(array)
    left_borders = [0]
    right_borders = [-1]

    # Находим точки, где есть выбросы
    for i in range(round(n*proc)):
        if hampel_array[i]:
            left_borders.append(i)

    for i in range(1, round(n*proc)):
        if hampel_array[-i]:
            right_borders.append(-i)

    # Формируем массивы длин отрезков
    delta_left = []
    for i in range(1, len(left_borders)):
        delta_left.append(left_borders[i] - left_borders[i-1])

    delta_right = []
    for i in range(1, len(right_borders)):
        delta_right.append(right_borders[i-1] - right_borders[i])

    # Находим максимальные длины плюс то что до
    left_border, right_border = max_len(delta_left, delta_right)

    return left_border, right_border


def remove_frame_by_outlier(img, proc):
    """
    Удаление рамки через выбросы. По усредненным массивам (полоса по ширине и полоса по высоте)
    определяются границы рамки.
    :param img: Исходное изображение;
    :param proc: Часть изображения, которую может занимать рамка с каждой стороны(0.0 до 1.0).
    :return: Изображение без рамки.
    """
    sh = img.shape
    h, w = sh[0], sh[1]
    img_array = img.sum(axis=2) / (255 * 3)

    delta = round(w*proc)
    array_w = np.zeros(w)
    for i in range(delta, h-delta, 10):
        array_w += img_array[i, :]

    delta = round(h * proc)
    array_h = np.zeros(h)
    for i in range(delta, w-delta, 10):
        array_h += img_array[:, i]

    left_border, right_border = where_frame_by_outlier(array_w, proc)
    top_border, bottom_border = where_frame_by_outlier(array_w, proc)

    return img[top_border:h-bottom_border, left_border:w-right_border, :]


def get_border_white_array(array, proc=0.20):
    """
    Пробегает с краев к центру элементы массива до первого темного элемента (< 245, Яркость от 0 до 255)
    :param array: Массив яркости;
    :param proc: Часть массива, которую достаточно рассмотреть с каждой стороны(0.0 до 1.0).
    :return: Левый и правый индекс в массиве.
    """
    delta = round(len(array) * proc)
    left_border = 0
    right_border = -1

    for i in range(5, delta):
        if array[i] > 245:
            left_border = i
        else:
            break

    for i in range(6, delta):
        if array[-i] > 245:
            right_border = -i
        else:
            break

    return left_border, right_border


def remove_white_back(img, proc=0.20):
    """
    Удаление белой рамки при сканировании.
    :param img: Исходное изображение;
    :param proc: Часть изображения, которую может занимать рамка с каждой стороны(0.0 до 1.0).
    :return: Изображение без белой рамки.
    """
    sh = img.shape
    h, w = sh[0], sh[1]
    img_array = img.sum(axis=2) / 3

    delta = round(w * proc)
    array_w = np.zeros(w)
    k = 0
    for i in range(delta, h - delta, 10):
        array_w += img_array[i, :]
        k += 1
    array_w = 1/k * array_w

    delta = round(h * proc)
    array_h = np.zeros(h)
    k = 0
    for i in range(delta, w - delta, 10):
        array_h += img_array[:, i]
        k += 1
    array_h = 1 / k * array_h

    left_border, right_border = get_border_white_array(array_w, proc)
    top_border, bottom_border = get_border_white_array(array_h, proc)

    return img[top_border:h+bottom_border, left_border:w+right_border, :]

