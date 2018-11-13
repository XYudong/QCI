# CNN with 1D data
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # select to use which GPU
import keras
from keras import models
from keras.models import Sequential

from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.utils import np_utils
from keras.layers import Input, Conv2D, BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import regularizers
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.preprocessing import image
from pyts.image import GASF, MTF, RecurrencePlots
keras.backend.set_image_data_format('channels_last')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
import pickle

np.random.seed(813306)


def readucr(path):
    # data = np.loadtxt(filename, delimiter=',')
    # print(data[0:100])
    data = pd.read_csv(path, header=None)
    data = np.array(data)
    # print(data.dtype)
    Y = data[:, 0]      # for .txt files
    X = data[:, 1:]
    # print(X[0:100])
    return X, Y


def loaddataset(dataset='ECG200'):
    # root = "../data/"
    # x_train, y_train = readucr(root+dataset+'/'+dataset+'/'+dataset+'_TRAIN.txt')
    # x_test, y_test = readucr(root+dataset+'/'+dataset+'/'+dataset+'_TEST.txt')
    root = '../data/'
    if dataset == 'ECG5000':
        fname_tr = 'ECG5000_class1_2_train.csv'
        fname_te = 'ECG5000_class1_2_test.csv'

        x, y = readucr(root + dataset + '/' + dataset + '/' + 'ECG5000_class1_2_all.csv')
        x_train, x_test,  y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=88)
    elif dataset == 'ECG200':
        fname_tr = 'ECG200_TRAIN.txt'
        fname_te = 'ECG200_TEST.txt'

        x_train, y_train = readucr(root + dataset + '/' + dataset + '/' + fname_tr)
        x_test, y_test = readucr(root + dataset + '/' + dataset + '/' + fname_te)
    else:
        print('invalid dataset name')
        return None
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
    img = np.resize(img, (img.shape[0], img.shape[1], 1))
    img_rgb = np.repeat(img.astype(np.float32), 3, axis=2)
    return img_rgb


def VGG_16_new():
    model = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    # Fine-tuning: freeze some layers
    for layer in model.layers[0:-8]:
        layer.trainable = False
    # for layer in model.layers:
    #     print(layer, layer.trainable)

    # rebuild the model
    # pool5 = Flatten(name='flatten')(model.outputs)        # this doesn't work!!
    pool5 = Flatten(name='flatten')(model.layers[-1].output)

    dense_1 = Dense(128, name='dense_1', kernel_regularizer=regularizers.l2(0.01))(pool5)
    bn_1 = BatchNormalization()(dense_1)    # normalize the inputs of nonlinear layer(activation layer)
    act_1 = Activation('relu')(bn_1)
    d1 = Dropout(0.5, name='drop1')(act_1)

    # dense_2 = Dense(10, name='dense_2', kernel_regularizer=regularizers.l2(0.01))(d1)
    # bn_2 = BatchNormalization()(dense_2)
    # act_2 = Activation('relu')(bn_2)
    # d2 = Dropout(0.5, name='drop2')(act_2)

    dense_3 = Dense(2, name='dense_3', kernel_regularizer=regularizers.l2(0.01))(d1)
    bn_3 = BatchNormalization()(dense_3)
    # prediction = Activation("softmax", name="softmax")(bn_3)
    prediction = Activation("sigmoid", name="sigmoid")(bn_3)  # for binary classificaction

    model_new = Model(inputs=model.inputs, outputs=prediction)

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


def simpleNN():
    img_input = Input(shape=(96,), name='input')
    x = Flatten(name='flatten')(img_input)
    x = Dense(50, name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dense(2, name='fc2')(x)
    x = BatchNormalization()(x)
    prediction = Activation('sigmoid', name='output')(x)

    model = Model(inputs=img_input, outputs=prediction)

    return model


def data_normalization(x_train, x_test):
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train_norm = (x_train - x_train_mean) / x_train_std
    x_test_norm = (x_test - x_train_mean) / x_train_std

    return x_train_norm, x_test_norm


def plt_acc_loss(hist):
    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(hist.history['acc'], c='dodgerblue', linewidth=2)
    plt.plot(hist.history['val_acc'], c='r')
    xlim = plt.gca().get_xlim()
    plt.plot(xlim, [0.9, 0.9], '--', c='seagreen')
    plt.ylim(0.3, 1.0)
    plt.grid(True)
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train acc', 'test acc ' + str(format(max(hist.history['val_acc']), '.3f'))], loc='lower right')

    # summarize history for loss
    # plt.figure(2, figsize=(8, 8))
    plt.subplot(212)
    plt.plot(hist.history['loss'], c='dodgerblue')
    plt.plot(hist.history['val_loss'], c='r')
    plt.grid(True)
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'test loss'], loc='upper right')
    # figname2 = 'vgg16_ECG200_loss_latest'
    # plt.savefig(figname2)
    # figname = 'vgg16_ECG200_latest'
    # plt.savefig(figname)

    return True


def prepare_data(pre_data, method):
    # transform the output from timeseries method to standard VGG input format
    x_rgb = []
    if method != 'comb':
        for img in pre_data:
            img_rgb = to_rgb(img)
            temp = cv2.resize(img_rgb, (224, 224)).astype(np.float32)
            x_rgb.append(temp)
    else:
        for i in range(len(pre_data[0])):
            a = cv2.resize(pre_data[0][i], (224, 224)).astype(np.float32)
            b = cv2.resize(pre_data[1][i], (224, 224)).astype(np.float32)
            c = cv2.resize(pre_data[2][i], (224, 224)).astype(np.float32)
            img_rgb = np.stack([a, b, c], axis=2)
            x_rgb.append(img_rgb)
    x_rgb = np.array(x_rgb)
    return x_rgb


def transform_to_2D(method, x_train):
    if method == 'gasf':
        gasf = GASF(image_size=x_train.shape[1] // 2, overlapping=False, scale=-1)
        x = gasf.fit_transform(x_train)
        print('applying GASF')
    elif method == 'mtf':
        mtf = MTF(image_size=x_train.shape[1], n_bins=4, quantiles='empirical', overlapping=False)
        x = mtf.fit_transform(x_train)
        print('applying MTF')
    elif method == 'rp':
        rp = RecurrencePlots(dimension=1, epsilon='percentage_points', percentage=10)
        x = rp.fit_transform(x_train)
        print('applying RP')
    else:
        print('wrong method')
        x = []

    return x


def transform_label(y):
    nb_classes = len(np.unique(y))
    # print("number of classes:", nb_classes)

    # transform raw class vector to integers from 0 to num_classes
    y = (y - y.min()) / (y.max() - y.min()) * (nb_classes - 1)
    # Converts a class vector (integers) to binary class matrix, because of the use of loss='categorical_crossentropy'.
    Y = np_utils.to_categorical(y, nb_classes)

    return Y


def kfold_idx(k, x_train, y_train):
    # # folds = [train_idx, val_idx]
    folds = StratifiedKFold(n_splits=k, random_state=55).split(x_train, y_train)
    print('Initiate '+str(k)+' folds')
    return folds


def before_train(x, y, method):
    print('start transforming ...')
    if method != 'comb':
        x_temp = transform_to_2D(method, x)
    else:
        x_temp = []
        temp0 = transform_to_2D('rp', x)
        x_temp.append(temp0)
        temp0 = transform_to_2D('gasf', x)
        x_temp.append(temp0)
        temp0 = transform_to_2D('mtf', x)
        x_temp.append(temp0)

    print('to RGB ...')
    x_rgb = prepare_data(x_temp, method)
    Y = transform_label(y)

    return x_rgb, Y


def lr_scheduler(epoch, lr):
    if epoch == 15:
        lr = lr * 0.5
    elif epoch == 30:
        lr = lr * 0.5
    return lr


def train_model(method='rp', arg_times=1, epochs=60, fname='ECG200'):
    x_train, y_train, x_test, y_test = loaddataset(fname)
    x_test, Y_test = before_train(x_test, y_test, method)

    hists = []  # histories
    x_test_rgb_all = []
    folds = kfold_idx(10, x_train, y_train)
    for j, (train_idx, val_idx) in enumerate(folds):
        print('\n')
        print('Cross-validation: ', j)
        x_val = x_train[val_idx]
        y_val = y_train[val_idx]
        x_tr, y_tr = white_noise_augmentation(x_train[train_idx], y_train[train_idx], arg_times)
        x_val, y_val = white_noise_augmentation(x_val, y_val, 3)

        # print('start transforming ...')
        # if method != 'comb':
        #     x_tr, x_te = transform_to_2D(method, x_train, x_test)
        # else:
        #     x_tr = []
        #     x_te = []
        #     temp0, temp1 = transform_to_2D('rp', x_train, x_test)
        #     x_tr.append(temp0), x_te.append(temp1)
        #     temp0, temp1 = transform_to_2D('gasf', x_train, x_test)
        #     x_tr.append(temp0), x_te.append(temp1)
        #     temp0, temp1 = transform_to_2D('mtf', x_train, x_test)
        #     x_tr.append(temp0), x_te.append(temp1)
        #
        # print('to RGB ...')
        # x_train_rgb = prepare_data(x_tr, method)
        # x_test_rgb = prepare_data(x_te, method)

        # (sample, row, column, channel), i.e.(100*arg_times,224,224,3)
        # print('train set:', x_train_rgb.shape)
        # print('test set:', x_test_rgb.shape)

        x_train_rgb, Y_train = before_train(x_tr, y_tr, method)
        x_val_rgb, Y_val = before_train(x_val, y_val, method)

        x_train_norm, x_test_norm = data_normalization(x_train_rgb, x_test)
        x_train_norm, x_val_norm = data_normalization(x_train_rgb, x_val_rgb)
        print('normalized training set:', x_train_norm.shape)
        print('normalized validation set:', x_val_norm.shape)

        # Y_train = transform_label(y_train)
        # Y_test = transform_label(y_test)

        # mini-batch gradient descent
        batch_size = min(int(x_train_norm.shape[0] / 10), 16)
        print('batch size: ', batch_size)

        # create a model
        model_new = VGG_16_new()
        # model_new = simpleNN()

        # two optimizers for choice
        adam = keras.optimizers.Adam(lr=0.0002)
        # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

        model_new.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        # callbacks
        reduce_lr = LearningRateScheduler(lr_scheduler)
        reduce_lr_plus = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)
        # # tensorboard = TensorBoard('logs/run_9')
        checkpointer = ModelCheckpoint('../weights/vgg16_ECG200_fold'+str(j)+'_txt.h5',
                                       monitor='val_acc', save_best_only=True)

        print("start training....")
        hist = model_new.fit(x_train_norm, Y_train, batch_size=batch_size, epochs=epochs,
                             verbose=2, validation_data=(x_val_norm, Y_val),
                             callbacks=[checkpointer, reduce_lr, reduce_lr_plus])

        # dump history dictionary
        with open('../history/vgg16_ECG200_fold'+str(j)+'_txt', 'w+b') as file:
            pickle.dump(hist.history, file)
        hists.append(hist)
        # Testing
        [loss, acc] = model_new.evaluate(x_test_norm, Y_test, batch_size=len(Y_test))
        print('TEST loss: ', loss,)
        print('TEST accuracy: ', acc)
        x_test_rgb_all.append(x_test_norm)

    # save all the normalized x_test for later evaluation
    x_test_rgb_all = np.array(x_test_rgb_all)
    np.save('../history/x_test_norm.npy', x_test_rgb_all)
    return hists
    # return True


def extractor(dataset='ECG200', method='comb'):
    model = VGG16(include_top=True, weights='imagenet')

    # img_input = Input(shape=(224, 224, 3), name='input')
    # x = Flatten(name='flatten')(img_input)
    # model_fea = Model(inputs=img_input, outputs=x)

    # dense_1 = Dense(128, name='dense_1')(model.get_layer(name='flatten').output)

    # Create a new model
    model_fea = Model(inputs=model.layers[0].input, outputs=model.get_layer(name='flatten').output)
    # or inputs=model.inputs is also ok

    # print('new model outputs: ', model_fea.outputs)
    # model_fea.summary()
    x_train, y_train, x_test, y_test = loaddataset(dataset)

    print('start transforming ...')
    if method != 'comb':
        x_tr, x_te = transform_to_2D(method, x_train, x_test)
    else:
        x_tr = []
        x_te = []
        temp0, temp1 = transform_to_2D('rp', x_train, x_test)
        x_tr.append(temp0), x_te.append(temp1)
        temp0, temp1 = transform_to_2D('gasf', x_train, x_test)
        x_tr.append(temp0), x_te.append(temp1)
        temp0, temp1 = transform_to_2D('mtf', x_train, x_test)
        x_tr.append(temp0), x_te.append(temp1)

    print('to RGB ...')
    x_train_rgb = prepare_data(x_tr, method)
    x_test_rgb = prepare_data(x_te, method)     # output array
    # x_test_rgb = []
    x_train_rgb, x_test_rgb = data_normalization(x_train_rgb, x_test_rgb)
    print(x_train_rgb.shape)     # (100,224,224,3) for ECG200
    print(x_test_rgb.shape)

    # fname1 = dataset + '_' + method + '_fc1_class1_2_train.csv'
    # fname2 = dataset + '_' + method + '_fc1_class1_2_test.csv'

    path = 'ECG200/'
    fname1 = dataset + '_' + method + '_100train_vgg.npy'
    fname2 = dataset + '_' + method + '_100test_vgg.npy'
    print('start extracting ...')
    for i in range(x_train_rgb.shape[0]):
        img = x_train_rgb[i]
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        # label
        label = [[y_train[i]]]
        # feature vector
        features = model_fea.predict(img)
        # print(fc1_features.shape)     #(1, 4096)
        temp = np.concatenate((label, features), axis=1)    # label-first
        if i == 0:
            ex = temp
        else:
            new = np.concatenate((ex, temp), axis=0)
            ex = new
        if i == x_train_rgb.shape[0] - 1:
            # df = pd.DataFrame(ex)
            # print('df shape of training: ', df.shape)
            # df.to_csv(path + fname1, mode='w+', header=None, index=None)
            print(ex.shape)
            np.save(path + fname1, ex)
    print('Features from Train: done')

    for i in range(x_test_rgb.shape[0]):
        img = x_test_rgb[i]
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        # label
        label = [[y_test[i]]]
        # feature vector
        features = model_fea.predict(img)
        # print(fc1_features.shape)     #(1, 4096)
        temp = np.concatenate((label, features), axis=1)
        if i == 0:
            ex = temp
        else:
            new = np.concatenate((ex, temp), axis=0)
            ex = new
        if i == x_test_rgb.shape[0] - 1:
            # df = pd.DataFrame(ex)
            # df.to_csv(path + fname2, mode='w+', header=None, index=None)
            print(ex.shape)
            np.save(path + fname2, ex)
    print('Features from Test: done')

    return True


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# # config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

t1 = time.time()

hists = train_model(method='comb', arg_times=3, epochs=50, fname='ECG200')
for i, hist in enumerate(hists):
    plt.figure(i+1, figsize=(8, 10))
    plt_acc_loss(hist)

# extractor('ECG200', 'comb')

# loaddataset()
t2 = time.time()
t = t2 - t1
print('This takes ' + str(t) + ' seconds.')

plt.show()
