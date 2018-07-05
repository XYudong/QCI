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
# import cv2
import numpy as np
from matplotlib import pyplot as plt
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
a = np.zeros((2,3))
c = np.zeros((2,3))
b = np.stack([a,c], axis=2)
print(b.shape)
