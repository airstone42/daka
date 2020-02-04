#!/usr/bin/env python3
import os
import re

import cv2
import numpy as np

data_path = 'cage/images/'


def extract() -> (np.ndarray, np.ndarray):
    l_data, r_data = [], []
    pattern = r'(\d)(\d)_\w.*\.png'
    for _, _, files in os.walk(data_path, topdown=False):
        for file in files:
            if 'png' not in file:
                continue
            match = re.match(pattern, file)
            if not match or len(match.groups()) != 2:
                continue

            l_num, r_num = match.groups()
            file = data_path + file
            img: np.ndarray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            height, width = img.shape
            l_part: np.ndarray = img[:height, :int(width / 2)]
            r_part: np.ndarray = img[:height, int(width / 2):]
            l_data.append({'image': l_part, 'number': l_num})
            r_data.append({'image': r_part, 'number': r_num})

    return l_data, r_data


def train():
    l_full, r_full = extract()
    np.random.shuffle(l_full)
    np.random.shuffle(r_full)
    xl_full, yl_full = np.array([x['image'] for x in l_full]), np.array([int(x['number']) for x in l_full])
    xr_full, yr_full = np.array([x['image'] for x in r_full]), np.array([int(x['number']) for x in r_full])
    xl_train, yl_train = xl_full[:int(0.4 * len(xl_full)) + 1], yl_full[:int(0.4 * len(yl_full)) + 1]
    xr_train, yr_train = xr_full[:int(0.4 * len(xr_full)) + 1], yr_full[:int(0.4 * len(yr_full)) + 1]
    xl_valid, yl_valid = xl_full[int(0.4 * len(xl_full)):int(0.8 * len(xl_full))], yl_full[int(0.4 * len(yl_full)):int(
        0.8 * len(yl_full))]
    xr_valid, yr_valid = xr_full[int(0.4 * len(xr_full)):int(0.8 * len(xr_full))], yr_full[int(0.4 * len(yr_full)):int(
        0.8 * len(yr_full))]
    xl_test, yl_test = xl_full[int(0.8 * len(xl_full)):], yl_full[int(0.8 * len(yl_full)):]
    xr_test, yr_test = xr_full[int(0.8 * len(xr_full)):], yr_full[int(0.8 * len(yr_full)):]

    pass


if __name__ == '__main__':
    extract()
    train()
