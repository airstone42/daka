#!/usr/bin/env python3
import os
import re

import cv2
import keras
import numpy as np

data_path = 'cage/images/'
model_path = 'data/model.h5'


def extract() -> list:
    data = []
    pattern = r'(\d\d)_\w.*\.png'
    for _, _, files in os.walk(data_path, topdown=False):
        for file in files:
            if 'png' not in file:
                continue
            match = re.match(pattern, file)
            if not match or len(match.groups()) != 1:
                continue

            num = match.groups()[0]
            file = data_path + file
            img: np.ndarray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            data.append({'image': img, 'number': num})

    return data


def train():
    full = extract()
    np.random.shuffle(full)
    x_full, y_full = np.array([x['image'] for x in full]), np.array([int(x['number']) for x in full])
    x_train, y_train = x_full[:int(len(x_full) / 3)], y_full[:int(len(y_full) / 3)]
    x_valid, y_valid = x_full[int(len(x_full) / 3):int(len(x_full) / 3 * 2)], y_full[int(len(y_full) / 3):int(
        len(y_full) / 3 * 2)]
    x_test, y_test = x_full[int(len(x_full) / 3 * 2):], y_full[int(len(y_full) / 3 * 2):]
    
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-7
    x_train = (x_train - x_mean) / x_std
    x_valid = (x_valid - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    x_train = x_train[..., np.newaxis]
    x_valid = x_valid[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print(model.evaluate(x_test, y_test))
        return

    model = keras.models.Sequential([
        keras.layers.Conv2D(128, 24, activation='relu', padding='same', input_shape=[70, 200, 1]),
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
        keras.layers.Dense(100, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))
    print(model.evaluate(x_test, y_test))
    model.save(model_path)


if __name__ == '__main__':
    train()
