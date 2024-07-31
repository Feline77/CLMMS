import utils_RRTS as turtle_tools
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

# Conversion of a binary sequence to a decimal sequence with a step size of three bits.
def bin2dec(secret):
    slice_bin = []

    slice_times = len(secret)//3

    for i in range(0,slice_times):
        slice_bin.append(secret[i*3:i*3+3])
    rest_bin_length = len(secret)-len(slice_bin)*3

    if rest_bin_length==1:
        rest_bin = [[0,0]+[secret[-1]]]
        slice_bin = slice_bin + rest_bin
    elif rest_bin_length==2:
        rest_bin = [[0]+secret[-2:]]
        slice_bin = slice_bin + rest_bin

    decimal_list = []
    for binary_value in slice_bin:
        binary_str = str(binary_value[0])+str(binary_value[1])+str(binary_value[2])
        decimal_value = int(binary_str,2)
        decimal_list.append(decimal_value)
    return decimal_list

# Conversion of a binary sequence to a decimal sequence
def dec2bin(decimal_list):
    binary_list = [format(num, '03b') for num in decimal_list]
    binary_string = ''.join(binary_list)
    binary_digits = [int(digit) for digit in binary_string]
    return binary_digits


def hiding(cover, secret, turtle_seed=6, turtle_size=256):
    # Conversion of a binary sequence to a decimal sequence
    dec_secret = bin2dec(secret)

    if np.ndim(cover) > 1:
        # cover is image
        cover_size = cover.shape[0]
        # convert image as sequence
        cover_reshape = cover.ravel()
    else:
        # cover is sequence
        cover_reshape = cover

    # Generate reference matrix and associated set
    ref_mat, ass_set = turtle_tools.get_turtle(turtle_seed, turtle_size)

    # initial action mat
    action_mat = []

    if len(dec_secret) > len(cover_reshape) // 2:
        print(f"The secret information is too long. The cover can only hide secret data with a length not exceeding {3 * len(cover_reshape) // 2} bits. Please use a larger cover image.")
        return None

    # initial stego
    else:
        stego_reshape = cover_reshape.copy()

        # hiding
        idx = 0
        for s in tqdm(dec_secret):
            pos = (cover_reshape[idx * 2] % turtle_size, cover_reshape[idx * 2 + 1] % turtle_size)

            # Retrieve the associated set of pos
            unit_set_ = [d.get(pos) for d in ass_set if pos in d]
            unit_set = [j for i in unit_set_ for j in i]
            # Retrieve the values of the turtle shell matrix corresponding to the associated set
            unit_dict = [{loc: int(ref_mat[loc[0], loc[1]])} for loc in unit_set]
            # Filter associated set with values equal to s
            filtered_dicts = [d for d in unit_dict if list(d.values())[0] == s]
            candidates = [list(d.keys())[0] for d in filtered_dicts]
            # Select the unit from candidate that is closest to pos
            hide_pos = min(candidates, key=lambda x: math.sqrt((x[0] - pos[0]) ** 2 + (x[1] - pos[1]) ** 2))
            # embedding
            stego_reshape[idx * 2] = turtle_size * (cover_reshape[idx * 2] // turtle_size) + hide_pos[0]
            stego_reshape[idx * 2 + 1] = turtle_size * (cover_reshape[idx * 2] // turtle_size) + hide_pos[1]
            idx += 1
            # Store the action operation
            action_mat.append((hide_pos[0] - pos[0], hide_pos[1] - pos[1]))

        if np.ndim(cover) > 1:
            stego = np.array(stego_reshape).reshape((cover_size, cover_size))
        else:
            stego = stego_reshape

        return stego, action_mat

def extracting(stego, action_mat, secret_len, turtle_seed=6, turtle_size=256):
    ref_mat, corr_set = turtle_tools.get_turtle(turtle_seed, turtle_size)

    if np.ndim(stego) > 1:
        stego_reshape = stego.ravel()
    else:
        stego_reshape = stego

    recovery_secret = []
    recovered_image_reshape = []
    recovered_image_reshape_ = stego_reshape.copy()

    embedding_times = secret_len // 3
    if secret_len % 3 == 1 or secret_len % 3 == 2:
        embedding_times += 1

    for idx in tqdm(range(embedding_times)):
        pos = (stego_reshape[idx * 2] % turtle_size, stego_reshape[idx * 2 + 1] % turtle_size)
        recovery_secret.append(ref_mat[pos[0], pos[1]])
        # 恢复秘密图像
        recovered_pos = [stego_reshape[idx * 2] // turtle_size + (pos[0] - action_mat[idx][0]), stego_reshape[idx * 2 + 1] // turtle_size + (pos[1] - action_mat[idx][1])]
        recovered_image_reshape += recovered_pos
    recovery_secret_ = dec2bin(recovery_secret)

    if len(recovery_secret_) != secret_len:
        rest_length = secret_len % 3
        recovery_secret = recovery_secret_[:3 * (secret_len // 3)] + recovery_secret_[-rest_length:]
    else:
        recovery_secret = recovery_secret_

    recovered_image_reshape_[:len(recovered_image_reshape)] = recovered_image_reshape
    recovered_image_reshape = recovered_image_reshape_

    if np.ndim(stego) > 1:
        recovered_image = recovered_image_reshape.reshape((stego.shape[0], stego.shape[0]))
    else:
        recovered_image = recovered_image_reshape

    return recovery_secret, recovered_image
