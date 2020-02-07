#!/usr/bin/env python3
import os
import re

import cv2
import keras
import numpy as np
import pandas as pd

DATA_PATH = 'cage/images/'
LEFT_PATH = 'data/left.h5'
RIGHT_PATH = 'data/right.h5'
NUM_PATH = 'data/numbers.csv'

DataSet = (np.ndarray, np.ndarray, np.ndarray)


def extract() -> (list, list):
    l_data, r_data = [], []
    pattern = r'(\d)(\d)_\w.*\.png'
    for _, _, files in os.walk(DATA_PATH, topdown=False):
        for file in files:
            if 'png' not in file:
                continue
            match = re.match(pattern, file)
            if not match or len(match.groups()) != 2:
                continue

            l_num, r_num = match.groups()
            file = DATA_PATH + file
            img: np.ndarray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            _, width = img.shape
            l_part: np.ndarray = img[..., :int(width / 2)]
            r_part: np.ndarray = img[..., int(width / 2):]
            l_data.append({'image': l_part, 'number': l_num})
            r_data.append({'image': r_part, 'number': r_num})

    return l_data, r_data


def load() -> (DataSet, DataSet):
    l_full, r_full = extract()
    np.random.shuffle(l_full)
    np.random.shuffle(r_full)
    xl_full, yl_full = np.array([x['image'] for x in l_full]), np.array([int(x['number']) for x in l_full])
    xr_full, yr_full = np.array([x['image'] for x in r_full]), np.array([int(x['number']) for x in r_full])

    xl_train, yl_train = xl_full[:int(len(xl_full) / 3)], yl_full[:int(len(yl_full) / 3)]
    xr_train, yr_train = xr_full[:int(len(xr_full) / 3)], yr_full[:int(len(yr_full) / 3)]
    xl_valid, yl_valid = \
        xl_full[int(len(xl_full) / 3):int(len(xl_full) / 3 * 2)], \
        yl_full[int(len(yl_full) / 3):int(len(yl_full) / 3 * 2)]
    xr_valid, yr_valid = \
        xr_full[int(len(xr_full) / 3):int(len(xr_full) / 3 * 2)], \
        yr_full[int(len(yr_full) / 3):int(len(yr_full) / 3 * 2)]
    xl_test, yl_test = \
        xl_full[int(len(xl_full) / 3 * 2):], \
        yl_full[int(len(yl_full) / 3 * 2):]
    xr_test, yr_test = \
        xr_full[int(len(xr_full) / 3 * 2):], \
        yr_full[int(len(yr_full) / 3 * 2):]

    xl_mean, xr_mean = \
        xl_train.mean(axis=0, keepdims=True), \
        xr_train.mean(axis=0, keepdims=True)
    xl_std, xr_std = \
        xl_train.std(axis=0, keepdims=True) + 1e-7, \
        xr_train.std(axis=0, keepdims=True) + 1e-7
    xl_train, xr_train = (xl_train - xl_mean) / xl_std, (xr_train - xr_mean) / xr_std
    xl_valid, xr_valid = (xl_valid - xl_mean) / xl_std, (xr_valid - xr_mean) / xr_std
    xl_test, xr_test = (xl_test - xl_mean) / xl_std, (xr_test - xr_mean) / xr_std
    xl_train, xr_train = xl_train[..., np.newaxis], xr_train[..., np.newaxis]
    xl_valid, xr_valid = xl_valid[..., np.newaxis], xr_valid[..., np.newaxis]
    xl_test, xr_test = xl_test[..., np.newaxis], xr_test[..., np.newaxis]

    if not os.path.exists(NUM_PATH):
        nums = {
            'l_mean': xl_mean.tolist(),
            'l_std': xl_std.tolist(),
            'r_mean': xr_mean.tolist(),
            'r_std': xr_std.tolist(),
        }
        with open(NUM_PATH, 'xt', encoding='utf-8', newline='\n') as f:
            pd.DataFrame([nums]).to_csv(f, index=False, line_terminator='\n')

    return ((xl_train, yl_train), (xl_valid, yl_valid), (xl_test, yl_test)), \
           ((xr_train, yr_train), (xr_valid, yr_valid), (xr_test, yr_test))


def train():
    l_full, r_full = load()
    (xl_train, yl_train), (xl_valid, yl_valid), (xl_test, yl_test) = l_full[0], l_full[1], l_full[2]
    (xr_train, yr_train), (xr_valid, yr_valid), (xr_test, yr_test) = r_full[0], r_full[1], r_full[2]

    if os.path.exists(LEFT_PATH) and os.path.exists(RIGHT_PATH):
        l_model = keras.models.load_model(LEFT_PATH)
        r_model = keras.models.load_model(RIGHT_PATH)
        print(l_model.evaluate(xl_test, yl_test))
        print(r_model.evaluate(xr_test, yr_test))
        return

    l_model = keras.models.Sequential([
        keras.layers.Conv2D(128, 24, activation='relu', padding='same', input_shape=[70, 100, 1]),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 12, activation='relu', padding='same'),
        keras.layers.Conv2D(256, 12, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(512, 12, activation='relu', padding='same'),
        keras.layers.Conv2D(512, 12, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    r_model = keras.models.clone_model(l_model)

    l_model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    l_model.fit(xl_train, yl_train, epochs=10, validation_data=(xl_valid, yl_valid))
    l_model.save(LEFT_PATH)
    r_model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    r_model.fit(xr_train, yr_train, epochs=10, validation_data=(xr_valid, yr_valid))
    r_model.save(RIGHT_PATH)

    print(l_model.evaluate(xl_test, yl_test))
    print(r_model.evaluate(xr_test, yr_test))


if __name__ == '__main__':
    train()
