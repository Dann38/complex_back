import os
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from numpy import linalg
from math import sqrt
import scipy.stats as st
import sys
import argparse
# python -m pip install openpyxl
# import pandas as pd


def get_pundas_array():
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_file', help="""     
    numpy файл с данными
    """)

    namespace = parser.parse_args()
    np_file = namespace.npy_file

    is_file = os.path.exists(np_file)
    if not is_file:
        print(np_file, " - Не удалось найти файл")
        sys.exit(1)
    with open(np_file, "rb") as f:
        data = np.load(f)
    data = pd.DataFrame(data)

    return data


def least_squares_reg(x, y):
    m = len(x)
    vec_right = np.zeros(m + 1)
    LSM = np.zeros((m + 1, m + 1))
    LSM[0, 0] = 1
    for i in range(0, m):
        xi_mean = np.mean(x[i])
        LSM[0, i + 1] = xi_mean
        LSM[i + 1, 0] = xi_mean
    for i in range(0, m):
        for j in range(i, m):
            xixj_mean = np.mean(x[i] * x[j])
            LSM[i + 1, j + 1] = xixj_mean
            LSM[j + 1, i + 1] = xixj_mean

    vec_right[0] = np.mean(y)
    for i in range(0, m):
        yxi = np.mean(x[i] * y)
        vec_right[i + 1] = yxi

    return linalg.solve(LSM, vec_right)


def regression_analysis(x, y, alpha=0.05, debug=True):
    m, n = np.shape(x)
    # ПОИСК КОЭФФИЦИЕНТОВ ===================
    b = least_squares_reg(x, y)

    # вывод ------------------------------
    if debug:
        print(f"y* = {b[0]:.3f}", end="")
        for i in range(1, len(b)):
            plus = "+" if b[i] > 0 else "-"
            print(f" {plus} {abs(b[i]):.2f}x_{i}", end="")
        print()
    # ======================================
    # Задаем функцию

    # Корреляционная матрица ================
    K = np.corrcoef([y, *x])
    xK = np.corrcoef(x)
    len_k = len(K)
    if debug:
        print("\n====Корреляционная матрица======")
        for i in range(len_k):
            for j in range(i + 1):
                print(f"{K[i, j]:5.2f}", end="\t")
            print()
        print("================================\n")
    # =======================================

    # Стандартное уравнение регрессии =======
    beta = np.zeros_like(b)
    # sigma_y = sqrt(y.var())
    sigma_y = sqrt(n * ((y ** 2).sum()) - y.sum() ** 2)
    for i in range(1, len(b)):
        # sigma_xi = sqrt(x[i-1].var())
        sigma_xi = sqrt(n * ((x[i - 1] ** 2).sum()) - x[i - 1].sum() ** 2)
        beta[i] = b[i] * sigma_xi / sigma_y
    if debug:
        print(f"t_y =", end="")
        for i in range(1, len(b)):
            plus = "+" if b[i] > 0 else "-"
            print(f" {plus} {abs(beta[i]):.2f}t_{i}", end="")
        print()
    # ========================================

    # Индекс множ. корреляции ================
    if debug:
        print("\n====Индекс множ. корреляции=====")

    R_yx = sqrt(sum(beta * K[0, :]))
    if debug:
        print(f"R_yx = {R_yx:5.2f}")
    # ----------------------------------------
    # R_yx = sqrt(1 - linalg.det(K)/linalg.det(xK)) #НЕ РАБОТАЕТ ДЛЯ 1 x
    # print(f"R_yx = {R_yx:5.2f}")
    # ========================================

    # Скорректированный коэффициент регрессии=
    if debug:
        print("\n====Скорр. коэф. регрессии=====")
    R2_hat = 1. - (1. - R_yx ** 2) * (n - 1) / (n - m - 1)
    if debug:
        print(f"R = {R2_hat:5.2f}")
    # ========================================

    # Суммы =================================
    y_model = np.ones(n) * b[0]
    for i in range(m):
        y_model = y_model + b[i + 1] * x[i]
    SS_R = ((y_model - y.mean()) ** 2).sum()
    SS_RES = ((y - y_model) ** 2).sum()
    SS_COM = ((y - y.mean()) ** 2).sum()
    if debug:
        print("====Суммы======================")
        print(f"""
    SS_R = {SS_R:.3f}
    SS_RES = {SS_RES:.3f}
    SS_COM = {SS_COM:.3f}
        """)
    # =======================================

    # Ошибки==================================
    X = np.array([np.ones(n), *x])
    Z_inv = linalg.inv(X @ X.T)
    S_pow_2 = SS_RES / (n - m - 1)
    Sb_z = []

    for i in range(len(X)):
        Sb_z.append(sqrt(S_pow_2 * Z_inv[i, i]))
    if debug:
        print(f"S^2 = {S_pow_2:.5f}")

    # ========================================

    # Проверка F критерия====================
    F_cr = False
    F_fact = R_yx ** 2 / (1 - R_yx ** 2) * (n - m - 1) / m
    F_tabl = st.f.ppf(1 - alpha, 1, n - m - 1)
    if F_fact > F_tabl:
        F_cr = True
    if debug:
        print("====Критерий Фишера==============")
        if F_cr:
            print(f"""
    F_fact > F_tabl
    F_fact = {F_fact:.3f} 
    F_tabl = {F_tabl:.3f}
            """)
        else:
            print(f"""
    F_fact < F_tabl
    F_fact = {F_fact:.3f} 
    F_tabl = {F_tabl:.3f}
            """)
            # ========================================

    # Критерий Стьюдента======================
    t_tabl = st.t.ppf(1 - alpha / 2, n - m - 1)
    T = []
    T_z = []
    for i in range(len(Sb_z)):
        # T.append(b[i+1]/Sb[i])
        T_z.append(b[i] / Sb_z[i])
    if debug:
        print("\n====Критерий Cтьюдент=============")
        print(f"t_табл = {t_tabl:.3f}")
    t_cr = True
    if debug:
        isNot = "" if abs(T_z[0]) > t_tabl else "НЕ"
        print(f"Свободный параметр статистически {isNot} значимый ({T_z[0]:.3f})")
    for i in range(1, m + 1):
        if abs(T_z[i]) > t_tabl:
            if debug:
                print(f"{i}-й параметр статистически значимый ({T_z[i]:.3f})")
        else:
            if debug:
                t_cr = False
                print(f"{i}-й параметр статистически НЕ значимый({T_z[i]:.3f})")
    # ========================================

    return R2_hat, F_cr, t_cr, b


def main():
    data = get_pundas_array()
    print(f"\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%% ВХОДНЫЕ ДАННЫЕ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(data)
    column = data.columns

    y = np.array(data[column[-1]])
    x = np.array(data[column[0:-1]]).T
    print(f"\n\n%%%%%%%%%%%%%%%%%%%%%%%%% МНОЖЕСТВЕННАЯ РЕГРЕССИЯ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    reg_data = regression_analysis(x, y)
    with open("coefficients_reg_model.dat", "w") as f:
        for b in reg_data[3]:
            f.write(str(b) + " ")


if __name__ == "__main__":
    main()