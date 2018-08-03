import pandas as pd
import numpy as np
import time


'''split the original data.csv into training and test data set'''


def load_data():
    # only return data with label 1 and 2
    data = pd.read_csv('../data/ECG5000/ECG5000/ECG5000_new.csv', header=None)
    # data = pd.read_csv('../data/ECG200/ECG200/ECG200_new.csv', header=None)
    print(data.shape)
    labels = np.array(data.iloc[:, -1])
    # print(type(labels))
    idx1 = [i for (i, l) in enumerate(labels.tolist()) if l == 1]
    idx2 = [i for (i, l) in enumerate(labels.tolist()) if l == 2]
    idx3 = [i for (i, l) in enumerate(labels.tolist()) if l == 3]
    idx4 = [i for (i, l) in enumerate(labels.tolist()) if l == 4]
    idx5 = [i for (i, l) in enumerate(labels.tolist()) if l == 5]
    print('idx(1,2,3,4,5): ', str([len(idx1),len(idx2),len(idx3),len(idx4),len(idx5),]))
    data = np.array(data.iloc[idx1, 0:])

    # data = np.array(data.iloc[:, 0:])   # label-last
    return data


def dump_data(data, type):
    df = pd.DataFrame(data)
    fname = '../data/ECG5000/ECG5000/ECG5000_class1_2_' + type + '.csv'
    # fname = '../data/ECG200/ECG200/ECG200_' + type + '.csv'
    df.to_csv(fname, mode='w+', header=None, index=None)
    # print('here')
    return True


data = load_data()
print('filtered data: ', data.shape)

# train_data = data[0:140, :]
# test_data = data[140:, :]
# train_batch = data[0:500, :]

# # store data in .csv file
# dump_data(train_data, 'train')
# dump_data(test_data, 'test')
# dump_data(data, 'all')


# xx = pd.read_csv('../data/ECG5000/ECG5000/ECG5000_class1_2_all.csv', header=None)
# # xx = pd.read_csv('../data/ECG200/ECG200/ECG200_train.csv', header=None)
# xx = np.array(xx)
# print(xx.shape)
# # print(xx[0])
# print(xx[:, -1])



