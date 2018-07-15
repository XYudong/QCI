import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def load_data(path):
    # load feature set
    data = pd.read_csv(path, header=None)
    y = np.array(data.iloc[:, -1])  # labels
    fea = np.array(data.iloc[:, 0:-1])
    return y, fea


def white_noise_augmentation(x, times=2):
    # augmentation of 1D data
    mu, sigma = 0, 0.1

    for i in range(0, times):
        noise = np.random.normal(mu, sigma, [x.shape[0], ])
        x1 = x + noise
    print(x1.shape)
    return x1


root = '../data/ECG5000/ECG5000/'
dataset = 'ECG5000_class1_2_train.csv'
y, fea = load_data(root + dataset)
print(fea.shape)
idx1 = list(y).index(1)
idx2 = list(y).index(2)
# print(idx1)

# print(type(fea))
class1 = fea[idx1, :]       # (140,)
class2 = fea[idx2, :]
t = range(len(class1))


class1_aug = white_noise_augmentation(class1)
class2_aug = white_noise_augmentation(class2)

f1 = plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t, class1)
plt.title('class 1 in ECG5000')
plt.subplot(2,1,2)
plt.plot(t, class2)
plt.title('class 2 in ECG5000')

f2 = plt.figure(2)
plt.subplot(2,1,1)
plt.plot(t, class1_aug)
plt.title('class 1 in ECG5000 with white noise')
plt.subplot(2,1,2)
plt.plot(t, class2_aug)
plt.title('class 2 in ECG5000 with white noise')

plt.show()


