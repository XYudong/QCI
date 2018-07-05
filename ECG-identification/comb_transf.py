# CNN with 1D data
import keras
from keras import models
from keras.models import Sequential

from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.utils import np_utils
from keras.layers import Input, Conv2D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing import image
from pyts.image import GASF, MTF, RecurrencePlots

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import cv2
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(813306)


def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    # print(data[0:100])
    Y = data[:, 0]
    X = data[:, 1:]
    # print(X[0:100])
    return X, Y


def loaddataset(fname='ECG200'):
    root = "../data/"
    x_train, y_train = readucr(root + fname + '/' + fname + '/' + fname + '_TRAIN.txt')
    x_test, y_test = readucr(root + fname + '/' + fname + '/' + fname + '_TEST.txt')
    return x_train, y_train, x_test, y_test


def white_noise_augmentation(x, y, times=1):
    # augmentation of 1D data
    mu, sigma = 0, 0.1

    for i in range(0, times):
        noise = np.random.normal(mu, sigma, [x.shape[0], x.shape[1]])
        x1 = x + noise
        x = np.concatenate((x, x1), axis=0)
        y = np.concatenate((y, y), axis=0)

        # print(x.shape, y.shape)
    return x, y


def to_rgb(img):
    # transform to rgb-style(i.e. 3 channels)
    img = np.resize(img, (*img.shape, 1))
    img_rgb = np.repeat(img.astype(np.float32), 3, axis=2)
    return img_rgb


def VGG_16_new():
    model = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    # Fine-tuning: freeze some layers
    for layer in model.layers[:-8]:
        layer.trainable = False
    # for layer in model.layers:
    #     print(layer, layer.trainable)

    # rebuild the model
    # pool5 = Flatten(name='flatten')(model.outputs)        # this doesn't work
    pool5 = Flatten(name='flatten')(model.layers[-1].output)

    dense_1 = Dense(128, name='dense_1', activation='relu')(pool5)
    d1 = Dropout(0.5)(dense_1)

    dense_2 = Dense(128, name='dense_2', activation='relu')(d1)
    d2 = Dropout(0.5)(dense_2)

    dense_3 = Dense(2, name='dense_3')(d2)
    # prediction = Activation("softmax", name="softmax")(dense_3)
    prediction = Activation("sigmoid", name="sigmoid")(dense_3)

    model_new = Model(inputs=model.layers[0].input, outputs=prediction)

    # # Create the model
    # model_new = models.Sequential()
    #
    # # Add the vgg convolutional base model
    # model_new.add(model)
    #
    # # Add new layers
    # model_new.add(Flatten())
    # model_new.add(Dense(128, activation='relu'))
    # model_new.add(Dropout(0.5))
    # model_new.add(Dense(128, activation='relu'))
    # model_new.add(Dropout(0.5))
    # model_new.add(Dense(2, activation='sigmoid'))
    #
    # model_new.summary()
    return model_new


def data_normalization(x_train, x_test):
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    return x_train, x_test


def plt_acc_loss(hist):
    # summarize history for accuracy
    plt.figure(1, figsize=(8, 8))
    plt.plot(hist.history['acc'], c='dodgerblue')
    plt.plot(hist.history['val_acc'], c='r')
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train acc', 'test acc'], loc='upper left')

    # summarize history for loss
    plt.figure(2, figsize=(8, 8))
    plt.plot(hist.history['loss'], c='dodgerblue')
    plt.plot(hist.history['val_loss'], c='r')
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'test loss'], loc='upper left')

    plt.show()
    return True


def prepare_data(x_tr1, x_tr2, x_tr3):
    # transform the output from timeseries-method to standard VGG input format
    x_rgb = []
    for i in range(len(x_tr1)):
        img = [x_tr1[i], x_tr2[i], x_tr3[i]]
        for j in range(3):
            img[j] = cv2.resize(img[j], (224, 224)).astype(np.float32)
        img_rgb = np.stack(img, axis=2)
        x_rgb.append(img_rgb)
    x_rgb = np.array(x_rgb)
    return x_rgb


def transform_to_2D(method, x_train):
    if method == 'gasf':
        gasf = GASF(image_size=x_train.shape[1] // 2, overlapping=False, scale=-1)
        x_tr = gasf.fit_transform(x_train)
        # x_te = gasf.fit_transform(x_test)
        print('applying GASF')
    elif method == 'mtf':
        mtf = MTF(image_size=x_train.shape[1], n_bins=4, quantiles='empirical', overlapping=False)
        x_tr = mtf.fit_transform(x_train)
        # x_te = mtf.fit_transform(x_test)
        print('applying MTF')
    elif method == 'rp':
        rp = RecurrencePlots(dimension=3, epsilon='percentage_points', percentage=10)
        x_tr = rp.fit_transform(x_train)
        # x_te = rp.fit_transform(x_test)
        print('applying RP')

    return x_tr


def transform_label(y):
    nb_classes = len(np.unique(y))
    # print("number of classes:", nb_classes)

    # transform raw class vector to integers from 0 to num_classes
    y = (y - y.min()) / (y.max() - y.min()) * (nb_classes - 1)
    # Converts a class vector (integers) to binary class matrix, because of the use of loss='categorical_crossentropy'.
    Y = np_utils.to_categorical(y, nb_classes)

    return Y


def train_model(method='gasf', arg_times=1, epochs=50, fname='ECG200'):
    x_train, y_train, x_test, y_test = loaddataset(fname)
    x_train, y_train = white_noise_augmentation(x_train, y_train, arg_times)

    x_tr, x_te = transform_to_2D(method, x_train, x_test)

    x_train_rgb = prepare_data(x_tr)
    x_test_rgb = prepare_data(x_te)

    # (sample, row, column, channel), i.e.(100*arg_times,224,224,3)
    # print('train set:', x_train_rgb.shape)
    # print('test set:', x_test_rgb.shape)

    x_train_rgb, x_test_rgb = data_normalization(x_train_rgb, x_test_rgb)
    print('normalized test set:', x_test_rgb.shape)

    Y_train = transform_label(y_train)
    Y_test = transform_label(y_test)

    batch_size = min(int(x_train_rgb.shape[0] / 10), 16)
    print('batch size: ', batch_size)

    # create a model
    model_new = VGG_16_new()

    # print(x_train_rgb.shape, Y_train.shape)
    # print(x_test_rgb.shape, Y_test.shape)
    # file = open('debug_data', 'a+')
    # file.write(str(Y_train)+'\n')
    # file.write(str(Y_test)+'\n')

    # two optimizers for choice
    adam = keras.optimizers.Adam(lr=0.001)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model_new.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.0001)

    print("start training....")
    hist = model_new.fit(x_train_rgb, Y_train, batch_size=batch_size, epochs=epochs,
                         verbose=2, validation_data=(x_test_rgb, Y_test), callbacks=[reduce_lr])
    model_new.save('../weights/vgg16_new.h5')

    return hist
    # return True


def extractor(fname='ECG200', method='mtf'):
    model = VGG16(include_top=True, weights='imagenet')
    # Create a new model in order to get the feature vector from FC1
    model_fea = Model(inputs=model.inputs, outputs=model.get_layer(name='fc1').output)

    # model.summary()
    x_train, y_train, x_test, y_test = loaddataset(fname)

    print('start transforming ...')
    x_tr1 = transform_to_2D('mtf', x_train)
    x_tr2 = transform_to_2D('gasf', x_train)
    x_tr3 = transform_to_2D('rp', x_train)

    print('to RGB ...')
    x_train_rgb = prepare_data(x_tr1, x_tr2, x_tr3)
    # x_test_rgb = prepare_data(x_te)     # output array
    x_test_rgb = []
    x_train_rgb, x_test_rgb = data_normalization(x_train_rgb, x_test_rgb)
    # print(x_train_rgb.shape)  # (100,224,224,3)

    file = open(fname + '_' + 'comb' + '_fc1_features_train.txt', 'a+')
    # file2 = open(fname + '_' + method + '_fc1_features_test.txt', 'a+')

    print('start predicting ...')
    for i in range(x_train_rgb.shape[0]):
        img = x_train_rgb[i]
        img = np.expand_dims(img, axis=0)  # (1,224,224,3)
        img = preprocess_input(img)
        # write labels
        file.write(str(y_train[i]) + ' ')
        # write feature vector
        fc1_features = model_fea.predict(img)
        for value in fc1_features[0]:
            file.write(str(value) + ' ')
        file.write('\n')
    file.close()

    # for i in range(x_test_rgb.shape[0]):
    #     img = x_test_rgb[i]
    #     img = np.expand_dims(img, axis=0)
    #     img = preprocess_input(img)
    #     file2.write(str(y_test[i])+' ')
    #     fc1_features = model_fea.predict(img)
    #     for value in fc1_features[0]:
    #         file2.write(str(value) + ' ')
    #     file2.write('\n')
    # file2.close()
    return True


config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# hist = train_model(method='mtf', arg_times=2, epochs=100, fname='ECG200')
# plt_acc_loss(hist)

extractor('ECG5000')






