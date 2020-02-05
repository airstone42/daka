#!/usr/bin/env python3
import ast
import logging
import os
import time

import cv2
import keras
import numpy as np
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

config_path = 'config.yml'
img_path = 'data/'
model_path = 'data/model.h5'
number_path = 'data/number.csv'

dk_url = 'https://dk.shmtu.edu.cn/'
cas_url = 'https://cas.shmtu.edu.cn/'
captcha_url = cas_url + 'cas/captcha'
checkin_url = dk_url + 'checkin'

df = pd.read_csv(number_path)
x_mean = np.array(ast.literal_eval(df['mean'][0]))
x_std = np.array(ast.literal_eval(df['std'][0]))
model = keras.models.load_model(model_path)


def recognize(img) -> str:
    img = (img - x_mean) / x_std
    img = img[..., np.newaxis]
    return str(model.predict_classes([img])[0]).zfill(2)
    pass


def login(s: requests.Session, r: requests.Response, config: dict) -> (bool, requests.Response):
    soup = BeautifulSoup(r.content, 'lxml')
    captcha = s.get(captcha_url)

    array = np.frombuffer(captcha.content, np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    valid_code = recognize(img)
    execution = soup.find('input', attrs={'type': 'hidden', 'name': 'execution'})['value']
    data = {
        'username': config['id'],
        'password': config['password'],
        'validateCode': valid_code,
        'execution': execution,
        '_eventId': 'submit',
        'geolocation': '',
    }

    post = s.post(r.url, data=data)
    soup = BeautifulSoup(post.content, 'lxml')
    logging.info('Login...')
    return (False, post) if soup.find('div', attrs={'class': 'alert alert-danger'}) else (True, post)


def checkin(s: requests.Session, r: requests.Response, config: dict) -> bool:
    if r.url != dk_url:
        return False

    soup = BeautifulSoup(r.content, 'lxml')
    check = str(soup.find('div', attrs={'class': 'form-group'}))
    flag = False
    if 'Health report have not been submitted today' in check:
        flag = False
    if 'Health report already submitted' in check:
        flag = True

    if not flag:
        data = {
            'xgh': config['id'],
            'region': config['region'],
            'rylx': config['contacted'],
            'status': config['health'],
        }
        post = s.post(checkin_url, data=data)
        logging.info('Checkin...')
        soup = BeautifulSoup(post.content, 'lxml')
        if 'Health report already submitted' in str(soup.find('div', attrs={'class': 'form-group'})):
            flag = True
    if flag:
        logging.info('Checkin successful!')

    return flag


def main():
    if not os.path.exists(config_path):
        print('Config does not exist!')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        if not config['id'] and config['number']:
            print('Config incorrect!')
    error_count = 0

    s = requests.Session()
    r = s.get(dk_url)

    if cas_url in r.url:
        result, r = login(s, r, config)
        while not result and error_count < 5:
            result, r = login(s, r, config)
            if result:
                break
            error_count += 1
            time.sleep(5)

    while not checkin(s, r, config) and error_count < 5:
        if checkin(s, r, config):
            break
        error_count += 1
        time.sleep(5)

    if not error_count < 5:
        logging.warning('Error!')


if __name__ == '__main__':
    main()
