import numpy as np
import pandas as pd


# 生成龟壳矩阵
def generate_tortoise(turtle_seed=0, size=256):
    """
    :param seed: 随机数种子，范围为0-7
    :param size: 龟壳矩阵大小
    :return:
    """
    tortoise = np.zeros((size, size), dtype=int)  # 初始化矩阵
    tortoise[0, 0] = turtle_seed

    col_regular = [2, 3] * (size // 2 + 1)
    col_regular = col_regular[:size - 1]

    # 生成第一列
    for idx in range(1, size):
        tortoise[idx, 0] = (tortoise[idx - 1, 0] + col_regular[idx - 1]) % 8
    # 根据列生成行
    for idx in range(1, size):
        tortoise[:, idx] = (tortoise[:, idx - 1] + 1) % 8

    return tortoise


# 获取龟壳矩阵中特殊单元的位置
def get_special_unit(N=256, M=256):
    # 首行
    if M % 2 == 0:
        pos_temp = list(range(0, M, 2)) + [M - 1]
        first_row = [(0, j) for j in pos_temp]
    else:
        pos_temp = list(range(0, M, 2))
        first_row = [(0, j) for j in pos_temp]
    # 首列
    if N % 4 == 0:
        pos_temp_ = sorted([0] + [i for i in range(3, N, 4)] + [i + 1 for i in range(3, N, 4)])
        pos_temp = [t for t in pos_temp_ if t < N]
        first_col = [(i, 0) for i in pos_temp]
    elif N % 4 == 1:
        pos_temp_ = sorted([0] + [i for i in range(3, N, 4)] + [i + 1 for i in range(3, N, 4)])
        pos_temp = [t for t in pos_temp_ if t < N]
        first_col = [(i, 0) for i in pos_temp]
    elif N % 4 == 2:
        pos_temp_ = sorted([0] + [i for i in range(3, N, 4)] + [i + 1 for i in range(3, N, 4)]) + [N - 1]
        pos_temp = [t for t in pos_temp_ if t < N]
        first_col = [(i, 0) for i in pos_temp]
    elif N % 4 == 3:
        pos_temp_ = sorted([0] + [i for i in range(3, N, 4)] + [i + 1 for i in range(3, N, 4)]) + [N - 2, N - 1]
        pos_temp = [t for t in pos_temp_ if t < N]
        first_col = [(i, 0) for i in pos_temp]
    # 末行
    if N % 4 == 0:
        if M % 2 == 0:
            pos_temp = list(range(0, M, 2)) + [M - 1]
        else:
            pos_temp = list(range(0, M, 2))
        last_row = [(N - 1, j) for j in pos_temp]
    elif N % 4 == 1:
        if M % 2 == 0:
            pos_temp = list(range(0, M, 2)) + [M - 1]
        else:
            pos_temp = list(range(0, M, 2))
        last_row = [(N - 2, j) for j in pos_temp]
        last_row += [(N - 1, j) for j in range(M)]
    elif N % 4 == 2:
        if M % 2 == 0:
            pos_temp = [0] + list(range(1, M, 2))
        else:
            pos_temp = [0] + list(range(1, M, 2)) + [M - 1]
        last_row = [(N - 1, j) for j in pos_temp]
    elif N % 4 == 3:
        if M % 2 == 0:
            pos_temp = [0] + list(range(1, M, 2))
        else:
            pos_temp = [0] + list(range(1, M, 2)) + [M - 1]
        last_row = [(N - 2, j) for j in pos_temp]
        last_row += [(N - 1, j) for j in range(M)]
    # 末列
    if M % 2 == 0:
        if N % 4 == 0:
            pos_temp_ = sorted([0] + [i for i in range(1, N, 4)] + [i + 1 for i in range(1, N, 4)] + [N - 1])
            pos_temp = [t for t in pos_temp_ if t < N]
            last_col = [(i, M - 1) for i in pos_temp]
        elif N % 4 == 1:
            pos_temp_ = sorted([0] + [i for i in range(1, N, 4)] + [i + 1 for i in range(1, N, 4)] + [N - 2, N - 1])
            pos_temp = [t for t in pos_temp_ if t < N]
            last_col = [(i, M - 1) for i in pos_temp]
        elif N % 4 == 2:
            pos_temp_ = sorted([0] + [i for i in range(1, N, 4)] + [i + 1 for i in range(1, N, 4)])
            pos_temp = [t for t in pos_temp_ if t < N]
            last_col = [(i, M - 1) for i in pos_temp]
        elif N % 4 == 3:
            pos_temp_ = sorted([0] + [i for i in range(1, N, 4)] + [i + 1 for i in range(1, N, 4)])
            pos_temp = [t for t in pos_temp_ if t < N]
            last_col = [(i, M - 1) for i in pos_temp]
    else:
        if N % 4 == 0:
            pos_temp_ = sorted([0] + [i for i in range(3, N, 4)] + [i + 1 for i in range(3, N, 4)])
            pos_temp = [t for t in pos_temp_ if t < N]
            last_col = [(i, M - 1) for i in pos_temp]
        elif N % 4 == 1:
            pos_temp_ = sorted([0] + [i for i in range(3, N, 4)] + [i + 1 for i in range(3, N, 4)])
            pos_temp = [t for t in pos_temp_ if t < N]
            last_col = [(i, M - 1) for i in pos_temp]
        elif N % 4 == 2:
            pos_temp_ = sorted([0] + [i for i in range(3, N, 4)] + [i + 1 for i in range(3, N, 4)] + [N - 1])
            pos_temp = [t for t in pos_temp_ if t < N]
            last_col = [(i, M - 1) for i in pos_temp]
        elif N % 4 == 3:
            pos_temp_ = sorted([0] + [i for i in range(3, N, 4)] + [i + 1 for i in range(3, N, 4)] + [N - 2, N - 1])
            pos_temp = [t for t in pos_temp_ if t < N]
            last_col = [(i, M - 1) for i in pos_temp]
    sepcial_unit = first_row + first_col + last_row + last_col
    return sepcial_unit


# 获取特殊单元的关联集
def get_special_set(special_unit, N, M):
    corr_set = []
    for unit in special_unit:
        row = unit[0]
        col = unit[1]
        ## 四个角
        # 左上角
        if row == 0 and col == 0:
            unit_set = [(row + 0, col + 0), (row + 0, col + 1), (row + 0, col + 2),
                        (row + 1, col + 0), (row + 1, col + 1), (row + 1, col + 2),
                        (row + 2, col + 0), (row + 2, col + 1), (row + 2, col + 2)]
            corr_set.append({unit: unit_set})

        # 右上角
        elif row == 0 and col == M - 1:
            unit_set = [(row + 0, col - 2), (row + 0, col - 1), (row + 0, col + 0),
                        (row + 1, col - 2), (row + 1, col - 1), (row + 1, col + 0),
                        (row + 2, col - 2), (row + 2, col - 1), (row + 2, col + 0)]
            corr_set.append({unit: unit_set})

        # 左下角
        elif row == N - 1 and col == 0:
            unit_set = [(row - 2, col + 0), (row - 2, col + 1), (row - 2, col + 2),
                        (row - 1, col + 0), (row - 1, col + 1), (row - 1, col + 2),
                        (row + 0, col + 0), (row + 0, col + 1), (row + 0, col + 2)]
            corr_set.append({unit: unit_set})

            # 右下角
        elif row == N - 1 and col == M - 1:
            unit_set = [(row - 2, col - 2), (row - 2, col - 1), (row - 2, col + 0),
                        (row - 1, col - 2), (row - 1, col - 1), (row - 1, col + 0),
                        (row + 0, col - 2), (row + 0, col - 1), (row + 0, col + 0)]
            corr_set.append({unit: unit_set})

            ## 四条边
        # 上边
        elif row == 0 and col != 0 and col != M - 1:
            unit_set = [(row + 0, col - 1), (row + 0, col + 0), (row + 0, col + 1),
                        (row + 1, col - 1), (row + 1, col + 0), (row + 1, col + 1),
                        (row + 2, col - 1), (row + 2, col + 0), (row + 2, col + 1)]
            corr_set.append({unit: unit_set})

        # 下边
        elif row == N - 1 and col != 0 and col != M - 1:
            unit_set = [(row - 2, col - 1), (row - 2, col + 0), (row - 2, col + 1),
                        (row - 1, col - 1), (row - 1, col + 0), (row - 1, col + 1),
                        (row + 0, col - 1), (row + 0, col + 0), (row + 0, col + 1)]
            corr_set.append({unit: unit_set})

            # 左边
        elif col == 0 and row != 0 and row != N - 1:
            unit_set = [(row - 1, col + 0), (row - 1, col + 1), (row - 1, col + 2),
                        (row + 0, col + 0), (row + 0, col + 1), (row + 0, col + 2),
                        (row + 1, col + 0), (row + 1, col + 1), (row + 1, col + 2)]
            corr_set.append({unit: unit_set})

            # 右边
        elif col == M - 1 and row != 0 and row != N - 1:
            unit_set = [(row - 1, col - 2), (row - 1, col - 1), (row - 1, col + 0),
                        (row + 0, col - 2), (row + 0, col - 1), (row + 0, col + 0),
                        (row + 1, col - 2), (row + 1, col - 1), (row + 1, col + 0)]
            corr_set.append({unit: unit_set})

            ## 其他
        else:
            unit_set = [(row - 1, col - 1), (row - 1, col + 0), (row - 1, col + 1),
                        (row + 0, col - 1), (row + 0, col + 0), (row + 0, col + 1),
                        (row + 1, col - 1), (row + 1, col + 0), (row + 1, col + 1)]
            corr_set.append({unit: unit_set})

    return corr_set


# 获取龟壳矩阵中常规单元的位置
def get_regular_unit(N=256, M=256):
    ## 获取顶点
    # 顶点所在行
    peak_row1 = list(range(0, N, 2))
    peak_row = [i for i in peak_row1 if i + 3 <= N - 1]
    # 顶点所在列
    peak_col = list(range(1, M - 1))
    peak_unit = []
    eve_row = list(range(0, len(peak_row), 2))
    odd_row = list(range(1, len(peak_row), 2))
    eve_col = list(range(0, len(peak_col), 2))
    odd_col = list(range(1, len(peak_col), 2))

    peak_unit1 = []
    for idx_row in eve_row:
        row = peak_row[idx_row]
        for idx_col in eve_col:
            col = peak_col[idx_col]
            peak_unit1 += [(row, col)]
    peak_unit2 = []
    for idx_row in odd_row:
        row = peak_row[idx_row]
        for idx_col in odd_col:
            col = peak_col[idx_col]
            peak_unit2 += [(row, col)]
    peak_unit = peak_unit1 + peak_unit2

    ## 根据顶点获取左上；右上；左下；右下；下顶；中上；中下；
    left_up_unit = []
    righ_up_unit = []
    left_do_unit = []
    righ_do_unit = []
    bottom_unit = []
    mid_up_unit = []
    mid_do_unit = []

    for unit in peak_unit:
        row = unit[0]
        col = unit[1]
        # 左上
        left_up_row = row + 1
        left_up_col = col - 1
        left_up_unit += [(left_up_row, left_up_col)]

        # 左下
        left_do_row = row + 2
        left_do_col = col - 1
        left_do_unit += [(left_do_row, left_do_col)]

        if col + 1 > M or col + 2 > M or col + 3 > M or col + 4 > M:
            # 右上
            righ_up_row = row + 1
            righ_up_col = col + 1
            righ_up_unit += [(righ_up_row, righ_up_col)]

            # 右下
            righ_do_row = row + 2
            righ_do_col = col + 1
            righ_do_unit += [(righ_do_row, righ_do_col)]

        if row == max([i[0] for i in peak_unit]):
            # 下顶
            bottom_row = row + 3
            bottom_col = col + 0
            bottom_unit += [(bottom_row, bottom_col)]

        # 中上
        mid_up_row = row + 1
        mid_up_col = col + 0
        mid_up_unit += [(mid_up_row, mid_up_col)]

        # 中下
        mid_do_row = row + 2
        mid_do_col = col + 0
        mid_do_unit += [(mid_do_row, mid_do_col)]

    return peak_unit, left_up_unit, left_do_unit, mid_up_unit, mid_do_unit, bottom_unit, righ_up_unit, righ_do_unit


def get_regular_set(*regular):
    corr_set = []
    # 上顶点
    for unit in regular[0]:
        a = unit[0]
        b = unit[1]
        unit_set = [(a, b), (a + 1, b - 1), (a + 2, b - 1), (a + 3, b), (a + 1, b + 1), (a + 2, b + 1), (a + 1, b),
                    (a + 2, b)]
        corr_set.append({unit: unit_set})

    # 左上
    for unit in regular[1]:
        a = unit[0]
        b = unit[1]
        unit_set = [(a, b), (a + 1, b), (a + 2, b + 1), (a, b + 2), (a + 1, b + 2), (a + 1, b + 1), (a, b + 1),
                    (a - 1, b + 1)]
        corr_set.append({unit: unit_set})
        # 左下
    for unit in regular[2]:
        a = unit[0]
        b = unit[1]
        unit_set = [(a, b), (a - 1, b), (a, b + 2), (a - 1, b + 2), (a + 1, b + 1), (a - 2, b + 1), (a, b + 1),
                    (a - 1, b + 1)]
        corr_set.append({unit: unit_set})

    # 中上
    for unit in regular[3]:
        a = unit[0]
        b = unit[1]
        unit_set = [(a, b), (a + 1, b), (a + 2, b), (a - 1, b), (a, b - 1), (a, b + 1), (a + 1, b - 1), (a + 1, b + 1)]
        corr_set.append({unit: unit_set})
    # 中下
    for unit in regular[4]:
        a = unit[0]
        b = unit[1]
        unit_set = [(a, b), (a - 2, b), (a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1), (a - 1, b - 1), (a - 1, b + 1)]
        corr_set.append({unit: unit_set})

    # 下顶点
    for unit in regular[5]:
        a = unit[0]
        b = unit[1]
        unit_set = [(a, b), (a - 1, b - 1), (a - 2, b - 1), (a - 3, b), (a - 1, b + 1), (a - 2, b + 1), (a - 1, b),
                    (a - 2, b)]
        corr_set.append({unit: unit_set})

    # 右上
    for unit in regular[6]:
        a = unit[0]
        b = unit[1]
        unit_set = [(a, b), (a + 1, b), (a, b - 2), (a + 1, b - 2), (a - 1, b - 1), (a + 2, b - 1), (a, b - 1),
                    (a + 1, b - 1)]
        corr_set.append({unit: unit_set})
    # 右下
    for unit in regular[7]:
        a = unit[0]
        b = unit[1]
        unit_set = [(a, b), (a - 1, b), (a, b - 2), (a - 1, b - 2), (a + 1, b - 1), (a - 2, b - 1), (a, b - 1),
                    (a - 1, b - 1)]
        corr_set.append({unit: unit_set})

    # 创建一个空字典用于存储每个键对应的值
    merged_dict = {}

    # 遍历输入列表
    for item in corr_set:
        for key, value in item.items():
            if key in merged_dict:
                # 如果键已经存在于字典中，则将值合并到对应的列表中
                merged_dict[key].extend(value)
            else:
                # 如果键不存在于字典中，则将键和对应的值添加到字典中
                merged_dict[key] = value

    # 构建最终的列表
    output_list = [{key: value} for key, value in merged_dict.items()]

    return output_list


# 汇总函数
def get_turtle(turtle_seed, size):
    tortoise = generate_tortoise(turtle_seed=turtle_seed, size=size)
    # 获取特殊元素的关联集
    special_unit = get_special_unit(size, size)
    sepcial_set = get_special_set(special_unit, size, size)
    # 获取常规元素的关联集
    peak_unit, left_up_unit, left_do_unit, mid_up_unit, mid_do_unit, bottom_unit, righ_up_unit, righ_do_unit = get_regular_unit(
        size, size)
    regular_set = get_regular_set(peak_unit, left_up_unit, left_do_unit, mid_up_unit, mid_do_unit, bottom_unit,
                                  righ_up_unit, righ_do_unit)
    corr_set = sepcial_set + regular_set
    return tortoise, corr_set