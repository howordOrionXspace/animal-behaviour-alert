#!usr/bin/env python
# encoding:utf-8
from __future__ import division




"""
Function: Parameter configuration module
"""




import os
import math
import json
import time




# GPU settings
train_gpu = [0]
# Category List
classes_path = "weights/classes.txt"
# input_shape
input_shape = [224, 224]
# Pre trained model weights
model_path = "weights/base.h5"
# Epochs settings
Init_Epoch = 0
Freeze_Epoch = 100
UnFreeze_Epoch = 200
# batchsize settings
Freeze_batch_size = 32
Unfreeze_batch_size = 32
Freeze_Train = True
# Initial learning rate
Init_lr = 1e-2
# Maximum learning rate
Min_lr = Init_lr * 0.01
lr_decay_type = "cos"
save_period = 10
# The folder where weights and log files are saved
save_dir = "runs/train/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
eval_flag = True
eval_period = 10
# Multi threaded data reading
num_workers = 1
# Early stop setting
ES = 100
# Dataset directory
datasetDir = "data/"
classes_list = os.listdir(datasetDir)
classes_list.sort()
print("classes_list: ", classes_list)
with open("weights/classes.txt", "w") as f:
    for one in classes_list:
        f.write(one.strip() + "\n")
