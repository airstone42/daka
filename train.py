#!/usr/bin/env python3
import os
import re

import cv2
import keras
import numpy as np

data_path = 'cage/images/'
left_path = 'data/left.h5'
right_path = 'data/right.h5'


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
    xl_train, yl_train = xl_full[:int(len(xl_full) / 3)], yl_full[:int(len(yl_full) / 3)]
    xr_train, yr_train = xr_full[:int(len(xr_full) / 3)], yr_full[:int(len(yr_full) / 3)]
    xl_valid, yl_valid = xl_full[int(len(xl_full) / 3):int(len(xl_full) / 3 * 2)], yl_full[int(len(yl_full) / 3):int(
        len(yl_full) / 3 * 2)]
    xr_valid, yr_valid = xr_full[int(len(xr_full) / 3):int(len(xr_full) / 3 * 2)], yr_full[int(len(yr_full) / 3):int(
        len(yr_full) / 3 * 2)]
    xl_test, yl_test = xl_full[int(len(xl_full) / 3 * 2):], yl_full[int(len(yl_full) / 3 * 2):]
    xr_test, yr_test = xr_full[int(len(xr_full) / 3 * 2):], yr_full[int(len(yr_full) / 3 * 2):]

    xl_mean, xr_mean = xl_train.mean(axis=0, keepdims=True), xr_train.mean(axis=0, keepdims=True)
    xl_std, xr_std = xl_train.std(axis=0, keepdims=True) + 1e-7, xr_train.std(axis=0, keepdims=True) + 1e-7
    xl_train, xr_train = (xl_train - xl_mean) / xl_std, (xr_train - xr_mean) / xr_std
    xl_valid, xr_valid = (xl_valid - xl_mean) / xl_std, (xr_valid - xr_mean) / xr_std
    xl_test, xr_test = (xl_test - xl_mean) / xl_std, (xr_test - xr_mean) / xr_std
    xl_train, xr_train = xl_train[..., np.newaxis], xr_train[..., np.newaxis]
    xl_valid, xr_valid = xl_valid[..., np.newaxis], xr_valid[..., np.newaxis]
    xl_test, xr_test = xl_test[..., np.newaxis], xr_test[..., np.newaxis]

    if os.path.exists(left_path) and os.path.exists(right_path):
        l_model = keras.models.load_model(left_path)
        r_model = keras.models.load_model(right_path)
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
    print(l_model.evaluate(xl_test, yl_test))
    l_model.save(left_path)
    r_model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    r_model.fit(xr_train, yr_train, epochs=10, validation_data=(xr_valid, yr_valid))
    print(r_model.evaluate(xr_test, yr_test))
    r_model.save(right_path)


if __name__ == '__main__':
    train()
