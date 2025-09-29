#!usr/bin/env python
# encoding:utf-8
from __future__ import division




"""
Function: Model development definition module

Develop and build models to build the overall calculation process

Development of Definition Based on Keras Implementation Model
"""




import os
import time
import json
from keras import backend as K
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    DepthwiseConv2D,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Reshape,
)
from keras.models import Model





def convBlock(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """
    Basic Block
    """
    filters = int(filters * alpha)
    x = Conv2D(
        filters, kernel, padding="same", use_bias=False, strides=strides, name="conv1"
    )(inputs)
    x = BatchNormalization(name="conv1_bn")(x)
    return Activation(relu6, name="conv1_relu")(x)




def dwConvBlock(
    inputs,
    pointwise_conv_filters,
    alpha,
    depth_multiplier=1,
    strides=(1, 1),
    block_id=1,
):
    """
    DDW Convolutional Block
    """
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = DepthwiseConv2D(
        (3, 3),
        padding="same",
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=False,
        name="conv_dw_%d" % block_id,
    )(inputs)

    x = BatchNormalization(name="conv_dw_%d_bn" % block_id)(x)
    x = Activation(relu6, name="conv_dw_%d_relu" % block_id)(x)
    x = Conv2D(
        pointwise_conv_filters,
        (1, 1),
        padding="same",
        use_bias=False,
        strides=(1, 1),
        name="conv_pw_%d" % block_id,
    )(x)
    x = BatchNormalization(name="conv_pw_%d_bn" % block_id)(x)
    return Activation(relu6, name="conv_pw_%d_relu" % block_id)(x)




def MobileNet(
    input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, classes=1000
):

    img_input = Input(shape=input_shape)
    x = convBlock(img_input, 32, alpha, strides=(2, 2))
    x = dwConvBlock(x, 64, alpha, depth_multiplier, block_id=1)
    x = dwConvBlock(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = dwConvBlock(x, 128, alpha, depth_multiplier, block_id=3)
    x = dwConvBlock(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = dwConvBlock(x, 256, alpha, depth_multiplier, block_id=5)
    x = dwConvBlock(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = dwConvBlock(x, 512, alpha, depth_multiplier, block_id=7)
    x = dwConvBlock(x, 512, alpha, depth_multiplier, block_id=8)
    x = dwConvBlock(x, 512, alpha, depth_multiplier, block_id=9)
    x = dwConvBlock(x, 512, alpha, depth_multiplier, block_id=10)
    x = dwConvBlock(x, 512, alpha, depth_multiplier, block_id=11)
    x = dwConvBlock(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = dwConvBlock(x, 1024, alpha, depth_multiplier, block_id=13)
    x = GlobalAveragePooling2D()(x)
    shape = (1, 1, int(1024 * alpha))
    x = Reshape(shape, name="reshape_1")(x)
    x = Dropout(dropout, name="dropout")(x)
    x = Conv2D(classes, (1, 1), padding="same", name="conv_preds")(x)
    x = Activation("softmax", name="act_softmax")(x)
    x = Reshape((classes,), name="reshape_2")(x)
    inputs = img_input
    model = Model(inputs, x, name="mobilenet_%0.2f" % (alpha))
    return model




def relu6(x):
    return K.relu(x, max_value=6)



