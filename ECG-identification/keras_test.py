import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
import time


def readucr(path):
    # data = np.loadtxt(filename, delimiter=',')

    data = pd.read_csv(path, header=None)      # more efficient than np.loadtxt()
    data = np.array(data)

    Y = data[:, 0]      # for .txt files
    X = data[:, 1:]
    # print(X[0:100])
    return X, Y


def load_dataset(dataset='ECG200'):
    root = '../data/'
    if dataset == 'ECG5000':
        fname_tr = 'ECG5000_class1_2_train.csv'
        fname_te = 'ECG5000_class1_2_test.csv'
        # fname_te = 'ECG5000_TEST.txt'
    elif dataset == 'ECG200':
        fname_tr = 'ECG200_TRAIN.txt'
        fname_te = 'ECG200_TEST.txt'
    else:
        print('invalid dataset name')
        return None
    x_test, y_test = readucr(root + dataset + '/' + dataset + '/' + fname_te)
    return x_test, y_test


# dataset = 'ECG5000'
# # x_te, y_te = load_dataset(dataset)
# # TODO: test the speed between pd.read_csv and np.load
#
#
# aa = np.ones((100, 25000), dtype=float)
# # print(type(aa))
# fname = 'test_data'
#
# # t1 = time.time()
# # np.save(fname, aa)
# # t2 = time.time()
# # print(t2-t1)
# #
# # t1 = time.time()
# # df = pd.DataFrame(aa)
# # df.to_csv(fname+'.csv', mode='w+', header=None, index=None)
# # t2 = time.time()
# # print(t2-t1)
#
# aa = np.array([1,2,3,4,5])
# a = [0,1,2]
#
# print(aa[a])

a = np.array([1,2,5,2,3])
idx = [i for i, x in enumerate(a) if x==2]
print(idx)
