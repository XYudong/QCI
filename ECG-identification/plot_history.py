import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle
from pyts.image import GASF, MTF, RecurrencePlots
import cv2

from keras.models import load_model


class History(object):
    def __init__(self, fname):
        self.fname = fname  # name of the history file
        self.hist = None    # dictionary of training history

    def draw_hist(self):
        for i in range(5, 6):
            self.load_hist()
            self.plt_acc_loss()

            path = './figures/latest/'
            plt.savefig(path + self.fname)

        plt.show()
        return None

    def load_hist(self):
        """load the dictionary of training history"""
        root = '../history/'
        with open(root + self.fname, 'rb') as infile:
            self.hist = pickle.load(infile, encoding='latin1')
        return self.hist

    def plt_acc_loss(self):
        # summarize history for accuracy
        plt.figure(figsize=(8, 10))
        plt.subplot(211)
        plt.plot(self.hist['acc'], c='dodgerblue', linewidth=2)
        plt.plot(self.hist['val_acc'], c='r')
        xlim = plt.gca().get_xlim()
        plt.plot(xlim, [0.9, 0.9], '--', c='seagreen')
        plt.ylim(0.3, 1.0)
        plt.grid(True)
        plt.title('Model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train acc', 'test acc ' + str(format(max(self.hist['val_acc']), '.3f'))], loc='lower right')

        # summarize history for loss
        plt.subplot(212)
        plt.plot(self.hist['loss'], c='dodgerblue')
        plt.plot(self.hist['val_loss'], c='r')
        plt.grid(True)
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'test loss'], loc='upper right')
        # figname = 'vgg16_ECG200_latest'
        # plt.savefig(fname)
        plt.tight_layout()

        return None


class Analysis(object):

    def __init__(self, model_name, dataset='ECG200'):
        self.model_name = model_name
        self.dataset = dataset

    def get_model(self):
        root = '../weights/'
        new_model = load_model(root + self.model_name)
        return new_model

    def readucr(self, path):
        # data = np.loadtxt(filename, delimiter=',')
        # print(data[0:100])
        data = pd.read_csv(path, header=None)
        data = np.array(data)
        Y = data[:, -1]
        X = data[:, 0:-1]
        # print(X[0:100])
        return X, Y

    def load_dataset(self):
        root = '../data/'
        if self.dataset == 'ECG5000':
            fname_tr = 'ECG5000_class1_2_train.csv'
            fname_te = 'ECG5000_class1_2_test.csv'
        elif self.dataset == 'ECG200':
            fname_tr = 'ECG200_train.csv'
            fname_te = 'ECG200_test.csv'
        else:
            print('invalid dataset name')
            return None
        x_train, y_train = self.readucr(root + self.dataset + '/' + self.dataset + '/' + fname_tr)
        x_test, y_test = self.readucr(root + self.dataset + '/' + self.dataset + '/' + fname_te)
        return x_train, y_train, x_test, y_test

    def transform_to_2D(self, method, x_train, x_test):
        if method == 'gasf':
            gasf = GASF(image_size=x_train.shape[1] // 2, overlapping=False, scale=-1)
            x_tr = gasf.fit_transform(x_train)
            x_te = gasf.fit_transform(x_test)
            print('applying GASF')
        elif method == 'mtf':
            mtf = MTF(image_size=x_train.shape[1], n_bins=4, quantiles='empirical', overlapping=False)
            x_tr = mtf.fit_transform(x_train)
            x_te = mtf.fit_transform(x_test)
            print('applying MTF')
        elif method == 'rp':
            rp = RecurrencePlots(dimension=3, epsilon='percentage_points', percentage=10)
            x_tr = rp.fit_transform(x_train)
            x_te = rp.fit_transform(x_test)
            print('applying RP')
        else:
            print('wrong method')
            x_tr = []
            x_te = []

        return x_tr, x_te

    def prepare_data(self, pre_data):
        # transform the output from timeseries method to standard VGG input format
        x_rgb = []
        for i in range(len(pre_data[0])):
            a = cv2.resize(pre_data[0][i], (224, 224)).astype(np.float32)
            b = cv2.resize(pre_data[1][i], (224, 224)).astype(np.float32)
            c = cv2.resize(pre_data[2][i], (224, 224)).astype(np.float32)
            img_rgb = np.stack([a, b, c], axis=2)
            x_rgb.append(img_rgb)
        x_rgb = np.array(x_rgb)
        return x_rgb

    def data_normalization(self, x_train, x_test):
        x_train_mean = x_train.mean()
        x_train_std = x_train.std()
        # x_train = (x_train - x_train_mean) / x_train_std
        x_test = (x_test - x_train_mean) / x_train_std
        return x_test

    def preprocess(self):
        x_train, y_train, x_test, y_test = self.load_dataset()
        print('start transforming ...')
        x_tr = []
        x_te = []
        temp0, temp1 = self.transform_to_2D('rp', x_train, x_test)
        x_tr.append(temp0), x_te.append(temp1)
        temp0, temp1 = self.transform_to_2D('gasf', x_train, x_test)
        x_tr.append(temp0), x_te.append(temp1)
        temp0, temp1 = self.transform_to_2D('mtf', x_train, x_test)
        x_tr.append(temp0), x_te.append(temp1)

        print('to RGB ...')
        x_train_rgb = self.prepare_data(x_tr)
        x_test_rgb = self.prepare_data(x_te)

        x_test_rgb = self.data_normalization(x_train_rgb, x_test_rgb)
        # print('normalized training set:', x_train_rgb.shape)
        print('normalized test set:', x_test_rgb.shape)
        return x_test_rgb, y_test

    def prediction(self):
        vgg_model = self.get_model()
        data, classes = self.preprocess()
        pred = vgg_model.predict(data)
        return pred, classes


model_name = 'vgg16_new_23.h5'
ana = Analysis(model_name, 'ECG200')

preds, labels = ana.prediction()
idx = []

for i in range(0, len(labels)):
    if labels[i] == -1:
        labels[i] = 0
# print(labels[0:10])
for pred in preds:
    # print(pred)
    # print(pred.shape)
    temp = np.argmax(pred)
    idx.append(temp)
# print(idx[0:10])

ll = labels - idx
FN = []
FP = []
true_pred = 0
FN = [i for i, x in enumerate(ll) if x == 1]
FP = [i for i, x in enumerate(ll) if x == -1]
false_pre = len(FN) + len(FP)

str(false_pre)

print('FN: ' + str(FN) + '; FP: ' + str(FP) + '; false_predictions: '+str(false_pre))



