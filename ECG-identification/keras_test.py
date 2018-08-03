import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random


def readucr(path):
    # data = np.loadtxt(filename, delimiter=',')
    # print(data[0:100])
    data = pd.read_csv(path, header=None)
    data = np.array(data)
    # print(data)
    Y = data[:, 0]      # for .txt files
    X = data[:, 1:]
    # print(X[0:100])
    return X, Y


def load_dataset(dataset='ECG200'):
    root = '../data/'
    if dataset == 'ECG5000':
        fname_tr = 'ECG5000_class1_2_train.csv'
        fname_te = 'ECG5000_class1_2_test.csv'
    elif dataset == 'ECG200':
        fname_tr = 'ECG200_TRAIN.txt'
        fname_te = 'ECG200_TEST.txt'
    else:
        print('invalid dataset name')
        return None
    x_test, y_test = readucr(root + dataset + '/' + dataset + '/' + fname_te)
    return x_test, y_test


# dataset = 'ECG200'
# x_te, y_te = load_dataset(dataset)
# print(x_te.shape)
# print(x_te[:, 1])
# print(y_te)

# x_9 = x_te[9, :]
# print(y_te[9])
#
# f1 = plt.figure(1)
# plt.plot(range(0, len(x_9)), x_9)
# plt.ylim(-3,5)
# plt.title('Ischemia signal with index(9) in test set')
#
# plt.show()

# t = [1,2,3,4,5]
# a = [[1,5,6,8,9], [2,3,3,3,3]]
# aa = np.array([[1,5,6,8,9], [2,3,3,3,3]])
# # b = random.sample(a, 3)
#
# print(a)
# plt.plot(aa.transpose())
# plt.show()

a = [1,2,3]
b = [1,2]
c = a+b
a.append(b)
print(c)
print(a)

