# CNN with 1D data
import keras
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Conv2D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from pyts.image import GASF, MTF, RecurrencePlots

import cv2
import numpy as np
from matplotlib import pyplot as plt



np.random.seed(813306)

def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    #print(data[0:100])
    Y = data[:,0]
    X = data[:,1:]
    #print(X[0:100])
    return X, Y

def loaddataset():
    root = "../data/"
    fname = 'ECG200'
    x_train, y_train = readucr(root+fname+'/'+fname+'/'+fname+'_TRAIN.txt')
    x_test, y_test = readucr(root+fname+'/'+fname+'/'+fname+'_TEST.txt')
    return x_train,y_train,x_test,y_test

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
    model = VGG16(include_top=False, input_shape=(224,224,3), weights='imagenet', pooling=None)
    # to freeze weights of feature extraction layers
    for i in [0,1, 3,4, 6,7,8, 10,11,12, 14,15,16]:
        model.layers[i].trainable = False

    # rebuild the model
    pool5 = Flatten(name='flatten')(model.outputs)

    dense_1 = Dense(128, name='dense_1', activation='relu')(pool5)
    d1 = Dropout(0.05)(dense_1)

    dense_2 = Dense(128, name='dense_2', activation='relu')(d1)
    d2 = Dropout(0.05)(dense_2)

    dense_3 = Dense(2, name='dense_3')(d2)
    # prediction = Activation("softmax", name="softmax")(dense_3)
    prediction = Activation("sigmoid", name="sigmoid")(dense_3)

    model_new = Model(inputs=model.inputs, outputs=prediction)

    #two optimizers for choice
    adam = keras.optimizers.Adam()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model_new.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model_new

def normalization(x_train,y_train,x_test,y_test):
    nb_classes = len(np.unique(y_test))
    print("number of classes:", nb_classes)

    # transform raw class vector to integers from 0 to num_classes
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    # Converts a class vector (integers) to binary class matrix, because of the use of loss='categorical_crossentropy'.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / (x_train_std)
    x_test = (x_test - x_train_mean) / (x_train_std)

    return x_train,Y_train,x_test,Y_test

def plt_acc_loss(hist):
    # summarize history for accuracy
    plt.figure(1, figsize=(8,8))
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.figure(2, figsize=(8, 8))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
    return True


def train_model(method ='gasf', arg_times=1, epochs=50):
    x_train,y_train,x_test,y_test = loaddataset()
    # x_train,y_train = white_noise_augmentation(x_train, y_train, arg_times)

    if method == 'gasf':
        gasf = GASF(image_size=x_train.shape[1]//2, overlapping=False, scale=-1)
        x_tr = gasf.fit_transform(x_train)
        x_te = gasf.fit_transform(x_test)
        print('applying GASF')
    elif method == 'mtf':
        mtf = MTF(image_size=x_train.shape[1], n_bins=4, quantiles='empirical', overlapping=False)
        x_tr = mtf.fit_transform(x_train)
        x_te = mtf.fit_transform(x_test)
        print('applying MTF')
    else:
        rp = RecurrencePlots(dimension=3, epsilon='percentage_points', percentage=10)
        x_tr = rp.fit_transform(x_train)
        x_te = rp.fit_transform(x_test)
        print('applying RP')

    x_train_rgb = []
    for img in x_tr:
        img_rgb = to_rgb(img)
        im_tr = cv2.resize(img_rgb, (224, 224)).astype(np.float32)
        x_train_rgb.append(im_tr)
    x_train_rgb = np.array(x_train_rgb)

    # (sample, row, column, channel), i.e.(100*arg_times,224,224,3)
    # print('train set:', x_train_rgb.shape)

    x_test_rgb = []
    for img in x_te:
        img_rgb = to_rgb(img)
        im_te = cv2.resize(img_rgb, (224, 224)).astype(np.float32)
        x_test_rgb.append(im_te)
    x_test_rgb = np.array(x_test_rgb)
    # print('test set:', x_test_rgb.shape)

    x_train_rgb, Y_train, x_test_rgb, Y_test = normalization(x_train_rgb, y_train, x_test_rgb, y_test)
    # print('normalized test set:', x_test_rgb.shape)
    batch_size = min(int(x_train_rgb.shape[0] / 10), 16)
    print(batch_size)
    model = VGG_16_new()    # already complied

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

    print("start training....")
    print(x_train_rgb.shape, Y_train.shape)
    print(x_test_rgb.shape, Y_test.shape)
    # file = open('debug_data', 'a+')
    # file.write(str(Y_train)+'\n')
    # file.write(str(Y_test)+'\n')

    hist = model.fit(x_train_rgb, Y_train, batch_size=batch_size, epochs=epochs,
                     verbose=2, validation_data=(x_test_rgb, Y_test))
    #
    # plt_acc_loss(hist)
    # return hist
    return True


hist = train_model(method='mtf', arg_times=1, epochs=50)






