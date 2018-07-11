# import keras
# from keras import models
# from keras.models import Sequential
#
# from keras.models import Model
# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
# from keras.utils import np_utils
# from keras.layers import Input, Conv2D
# from keras.layers.core import Flatten, Dense, Dropout, Activation
# from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
# from keras.optimizers import SGD
# from keras.callbacks import ReduceLROnPlateau
# from keras.preprocessing import image
#
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
#
# model = VGG16(include_top=True, weights='imagenet')
# model_new = VGG16(include_top=False, input_shape=(224,224,3), weights='imagenet')
#
# print(model.inputs)
# print(model.layers[0].input)


# a = dict(mtf=[2,3,4], gasf=[1,2,3])
# print(a)
# a['rp'] = [1,2,3,4,5]
# print(a)
# a = np.zeros((2,3))
# c = np.zeros((3,3))
# # b = np.stack([a,c], axis=2)
# b = [c]
# b.append(a)
# b.append(np.ones((2,4)))

# def test():
#     a = np.array([2,3])
#     b = np.array([1,2,1])
#
#     return a,b
#
# test()

a = np.array([1.,2,3,4]).astype(np.float32)
b = np.expand_dims(a, axis=0)
print(b.shape)


# df = pd.DataFrame(b)
# df.to_csv('a_test.csv', mode='w+', header=None, index=None)
#
# c = np.array([1,1,1])
# # df2 = pd.DataFrame(c)
# # df2.to_csv('a_test.csv', mode='a+', header=None, index=None)
# #
# data = pd.read_csv('a_test.csv', header=None)
#
# # # print(a.shape)
# print(data)
# # print(data.as_matrix())
# print(data.values.shape)
# values = data.values
# print(values[0])

aa = np.ones((1,10))
bb = [[2]]

cc = np.concatenate((bb,aa), axis=1)
dd = cc
print(cc)
print(type(cc))
xx = np.array([1,2,3,3])
print(type(xx))
cc = [[]]
# ee = np.concatenate((cc,dd), axis=0)
# ee = dd.append([[]], axis)
# print(ee)

