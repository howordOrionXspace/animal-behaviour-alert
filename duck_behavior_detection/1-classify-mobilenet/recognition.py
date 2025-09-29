#!usr/bin/env python
# encoding:utf-8
from __future__ import division





"""
功能：   模型实体化推理计算模块
        加载训练好的模型权重实现输入图像数据的计算 
"""






import os
import json
import time
import numpy as np
from utils.utils import *
from mobilenet import MobileNet
import matplotlib.pyplot as plt





class recognitionModel(object):
    _defaults = {
        "model_path": "runs/train/best.h5",
        "classes_path": "weights/classes.txt",
        "input_shape": [224, 224],
    }



    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"



    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()



    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(".h5"), "Keras model or weights must be a .h5 file."
        self.model = MobileNet(
            input_shape=[self.input_shape[0], self.input_shape[1], 3],
            classes=self.num_classes,
            alpha=1.0,
        )
        self.model.load_weights(self.model_path)
        print("{} model, and classes loaded.".format(model_path))



    def recognitionImage(self, image):
        image = cvtColor(image)
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        image_data = np.expand_dims(
            preprocess_input(np.array(image_data, np.float32)), 0
        )
        preds = self.model.predict(image_data)[0]
        pred_label = self.class_names[np.argmax(preds)]
        probability = np.max(preds)
        pred_proba = float(probability)
        return pred_label, pred_proba



