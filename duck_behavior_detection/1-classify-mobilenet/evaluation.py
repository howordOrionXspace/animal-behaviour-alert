#!usr/bin/env python
# encoding:utf-8
from __future__ import division




"""
Function: Model testing and evaluation module
"""





import os
import json
import time
import math
import random
import numpy as np
from recognition import *
from utils.utils import *
from utils.metrics import *
from config import *
from PIL import Image
from sklearn.metrics import *





# Load Dataset
testDir = "runs/test/"
if not os.path.exists(testDir):
    os.makedirs(testDir)
# Load parsing dataset
with open("dataset.json") as f:
    dataset = json.load(f)
train_dataset = dataset["train"]
test_dataset = dataset["test"]
print("train_dataset_size: ", len(train_dataset))
print("test_dataset_size: ", len(test_dataset))





class Evaluation(recognitionModel):
    """
    Model evaluation calculation
    """

    def imageInfer(self, image):
        image = cvtColor(image)
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        image_data = np.expand_dims(
            preprocess_input(np.array(image_data, np.float32)), 0
        )
        preds = self.model.predict(image_data)[0]
        return preds




if __name__ == "__main__":


    print(
        "=====================================Start Evaluation Model====================================="
    )


    eva = Evaluation()
    evaluteModel(eva, dataset["test"], testDir)


