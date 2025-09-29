#!usr/bin/env python
# encoding:utf-8
from __future__ import division




"""
Function: Model training calculation module

Developing and constructing an action recognition model based on manually classified action datasets
"""




import os
import json
import time
import math
import random
import datetime
import numpy as np
from keras.callbacks import *
from keras.layers import Conv2D, Dense, DepthwiseConv2D
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.dataloader import ClsDatasets
from utils.utils import get_classes, get_lr_scheduler
from config import *
from mobilenet import MobileNet



from keras import backend as K
# Set TensorFlow as backend
K.set_image_dim_ordering("tf")
# Specify GPU operation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# Configure using 90% CPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.50
session = tf.Session(config=config)
# Set session
KTF.set_session(session)





if __name__ == "__main__":


    print(
        "=====================================Start Training Classification Model====================================="
    )


    # 获取类别
    class_names, num_classes = get_classes(classes_path)
    # 创建分类模型
    model = MobileNet(
        input_shape=[input_shape[0], input_shape[1], 3], classes=num_classes, alpha=1.0
    )
    print("Load weights {}.".format(model_path))
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    # 加载解析数据集
    with open("dataset.json") as f:
        dataset = json.load(f)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    print("train_dataset_size: ", len(train_dataset))
    print("test_dataset_size: ", len(test_dataset))
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    np.random.seed(10101)
    np.random.shuffle(train_dataset)
    np.random.seed(None)
    # 权重加载
    if Freeze_Train:
        freeze_layers = 81
        for i in range(freeze_layers):
            model.layers[i].trainable = False
        print(
            "Freeze the first {} layers of total {} layers.".format(
                freeze_layers, len(model.layers)
            )
        )
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
    start_epoch = Init_Epoch
    end_epoch = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch
    nbs = 64
    lr_limit_max = 1e-1
    lr_limit_min = 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(
        max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2
    )
    # 模型构建训练
    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(lr=Init_lr_fit, momentum=0.9, nesterov=True),
        metrics=["categorical_accuracy"],
    )
    lr_scheduler_func = get_lr_scheduler(
        lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch
    )
    train_dataloader = ClsDatasets(
        train_dataset, input_shape, batch_size, num_classes, train=True
    )
    val_dataloader = ClsDatasets(
        test_dataset, input_shape, batch_size, num_classes, train=False
    )
    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    # 训练记录
    logDir = save_dir + "loss_" + str(time_str) + "/"
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    logging = TensorBoard(logDir)
    loss_history = LossHistory(logDir)
    checkpoint = ModelCheckpoint(
        save_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
        monitor="val_loss",
        save_weights_only=True,
        save_best_only=False,
        period=save_period,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=10, verbose=1
    )
    lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)
    callbacks = [logging, loss_history, checkpoint, lr_scheduler]
    # 训练拟合
    if start_epoch < end_epoch:
        print(
            "Train on {} samples, val on {} samples, with batch size {}.".format(
                train_size, test_size, batch_size
            )
        )
        model.fit_generator(
            generator=train_dataloader,
            steps_per_epoch=train_size // batch_size,
            validation_data=val_dataloader,
            validation_steps=test_size // batch_size,
            epochs=end_epoch,
            initial_epoch=start_epoch,
            use_multiprocessing=True if num_workers > 1 else False,
            workers=num_workers,
            callbacks=callbacks,
        )
    if Freeze_Train:
        batch_size = Unfreeze_batch_size
        start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
        end_epoch = UnFreeze_Epoch
        nbs = 64
        lr_limit_max = 1e-1
        lr_limit_min = 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(
            max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2
        )
        lr_scheduler_func = get_lr_scheduler(
            lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch
        )
        lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)
        callbacks = [logging, loss_history, checkpoint, lr_scheduler]
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(
            loss="categorical_crossentropy",
            optimizer=SGD(lr=Init_lr_fit, momentum=0.9, nesterov=True),
            metrics=["categorical_accuracy"],
        )
        train_dataloader.batch_size = Unfreeze_batch_size
        val_dataloader.batch_size = Unfreeze_batch_size
        print(
            "Train on {} samples, val on {} samples, with batch size {}.".format(
                train_size, test_size, batch_size
            )
        )
        model.fit_generator(
            generator=train_dataloader,
            steps_per_epoch=train_size // batch_size,
            validation_data=val_dataloader,
            validation_steps=test_size // batch_size,
            epochs=end_epoch,
            initial_epoch=start_epoch,
            use_multiprocessing=True if num_workers > 1 else False,
            workers=num_workers,
            callbacks=callbacks,
        )
