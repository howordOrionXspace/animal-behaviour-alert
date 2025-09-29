#!usr/bin/env python
# encoding:utf-8
from __future__ import division



"""
功能：   模型训练计算所需的数据加载器
        实现数据加载逻辑
"""




import os
import cv2
import time
import json
import math
import keras
import numpy as np
from keras.utils import np_utils
from PIL import Image
from random import shuffle
from .utils import cvtColor, preprocess_input




class ClsDatasets(keras.utils.Sequence):
    """
    数据加载器
    """

    def __init__(self, annotation_lines, input_shape, batch_size, num_classes, train):
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train = train


    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))


    def __getitem__(self, index):
        X_train = []
        Y_train = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.length
            annotation_path = self.annotation_lines[i][1]
            image = Image.open(annotation_path)
            image = self.get_random_data(image, self.input_shape, random=self.train)
            image = preprocess_input(np.array(image).astype(np.float32))
            X_train.append(image)
            Y_train.append(int(self.annotation_lines[i][0]))
        X_train = np.array(X_train)
        Y_train = np_utils.to_categorical(
            np.array(Y_train), num_classes=self.num_classes
        )
        return X_train, Y_train


    def on_epoch_end(self):
        shuffle(self.annotation_lines)


    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a


    def get_random_data(
        self, image, input_shape, jitter=0.3, hue=0.1, sat=0.7, val=0.3, random=True
    ):
        image = cvtColor(image)
        iw, ih = image.size
        h, w = input_shape
        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            return image_data
        new_ar = (
            iw
            / ih
            * self.rand(1 - jitter, 1 + jitter)
            / self.rand(1 - jitter, 1 + jitter)
        )
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        rotate = self.rand() < 0.5
        if rotate:
            angle = np.random.randint(-15, 15)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(
                np.array(image), M, (w, h), borderValue=[128, 128, 128]
            )
        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
        )
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
