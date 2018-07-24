import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from pyts.image import GASF, MTF, RecurrencePlots


class Draw(object):
    def __init__(self, root, dataset):
        self.root = root
        self.dataset = dataset

    def load_data(self, path):
        # load feature set
        data = pd.read_csv(path, header=None)
        y = np.array(data.iloc[:, -1])  # labels
        fea = np.array(data.iloc[:, 0:-1])
        return y, fea

    def white_noise_augmentation(self, x, times=2):
        # augmentation of 1D data
        mu, sigma = 0, 0.1

        for i in range(0, times):
            noise = np.random.normal(mu, sigma, [x.shape[0], ])
            x1 = x + noise
        print(x1.shape)
        return x1

    def transform_to_2D(self, method, x_train):
        if method == 'gasf':
            gasf = GASF(image_size=x_train.shape[1] // 2, overlapping=False, scale=-1)
            x_tr = gasf.fit_transform(x_train)
            print('applying GASF')
        elif method == 'mtf':
            mtf = MTF(image_size=x_train.shape[1], n_bins=4, quantiles='empirical', overlapping=False)
            x_tr = mtf.fit_transform(x_train)
            print('applying MTF')
        elif method == 'rp':
            rp = RecurrencePlots(dimension=1, epsilon='percentage_points', percentage=10)
            x_tr = rp.fit_transform(x_train)
            print('applying RP')
        else:
            print('wrong method')
            x_tr = []

        return x_tr

    def plot_signal(self):
        y, fea = self.load_data(self.root + self.dataset)
        print(fea.shape)
        idx1 = list(y).index(1)
        idx2 = list(y).index(2)
        # print(idx1)

        # print(type(fea))
        class1 = fea[idx1, :]  # (140,)
        class2 = fea[idx2, :]
        t = range(len(class1))

        class1_aug = self.white_noise_augmentation(class1)
        class2_aug = self.white_noise_augmentation(class2)

        f1 = plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(t, class1)
        plt.title('class 1 in ECG5000')
        plt.subplot(2, 1, 2)
        plt.plot(t, class2)
        plt.title('class 2 in ECG5000')

        f2 = plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(t, class1_aug)
        plt.title('class 1 in ECG5000 with white noise')
        plt.subplot(2, 1, 2)
        plt.plot(t, class2_aug)
        plt.title('class 2 in ECG5000 with white noise')

    def plot_image(self):
        y, fea = self.load_data(self.root + self.dataset)
        print('feature shape: ', fea.shape)
        idx1 = list(y).index(1)     # index of the first occurrence
        idx2 = list(y).index(2)

        # print(fea[1,:].shape)     # (140,)
        class1 = fea[idx1, :].reshape(1, -1)
        class2 = fea[idx2, :].reshape(1, -1)

        # to image
        im_rp1 = self.transform_to_2D('rp', class1)
        im_rp2 = self.transform_to_2D('rp', class2)
        im_gasf1 = self.transform_to_2D('gasf', class1)
        im_gasf2 = self.transform_to_2D('gasf', class2)
        im_mtf1 = self.transform_to_2D('mtf', class1)
        im_mtf2 = self.transform_to_2D('mtf', class2)

        # print(im_rp1[0].shape)
        f1 = plt.figure(1)
        plt.subplot(2,1,1)
        plt.imshow(im_rp1[0], cmap='binary')
        plt.title('class1_RP')
        plt.subplot(2,1,2)
        plt.imshow(im_rp2[0], cmap='binary')
        plt.title('class2_RP')

        f2 = plt.figure(2)
        plt.subplot(2,1,1)
        plt.imshow(im_gasf1[0], cmap='binary')
        plt.title('class1_GASF')
        plt.subplot(2,1,2)
        plt.imshow(im_gasf2[0], cmap='binary')
        plt.title('class2_GASF')

        f3 = plt.figure(3)
        plt.subplot(2,1,1)
        plt.imshow(im_mtf1[0], cmap='binary')
        plt.title('class1_MTF')
        plt.subplot(2,1,2)
        plt.imshow(im_mtf2[0], cmap='binary')
        plt.title('class2_MTF')
        return None


root = '../data/ECG5000/ECG5000/'
dataset = 'ECG5000_class1_2_train.csv'

drawing = Draw(root, dataset)
drawing.plot_image()

# plt.tight_layout()
plt.show()


