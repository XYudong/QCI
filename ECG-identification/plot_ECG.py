import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random

from pyts.image import GASF, MTF, RecurrencePlots


class Draw(object):
    def __init__(self, root, dataset):
        self.root = root
        self.dataset = dataset

    def load_data(self, path):
        # load feature set
        data = pd.read_csv(path, header=None)
        y = np.array(data.iloc[:, 0])  # labels
        fea = np.array(data.iloc[:, 1:])
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
        # idx1 = list(y).index(1)
        # idx2 = list(y).index(2)
        # print(idx1)

        idx1 = [i for i, label in enumerate(y) if label == 1]
        idx2 = [i for i, label in enumerate(y) if label == -1]
        # idx number in separate class set of the TEST set
        ms = random.sample(range(0, len(idx1)), int(np.floor(0.5*len(idx1))))
        ns = random.sample(range(0, len(idx2)), int(np.floor(0.7*len(idx2))))

        m = [92]        # idx number in the TEST set
        n = [55, 75]
        print('ms: ', len(ms))
        print('ns: ', len(ns))

        class1 = []     # normal
        class2 = []     # abnormal
        for i in m:
            class1.append(fea[i, :])
        for j in n:
            class2.append(fea[j, :])

        class1s = []     # normal
        class2s = []     # abnormal
        for i in ms:
            class1s.append(fea[idx1[i], :])  # (140,)
        for j in ns:
            class2s.append(fea[idx2[j], :])
        t = range(len(class1[0]))

        # class1_aug = self.white_noise_augmentation(class1)
        # class2_aug = self.white_noise_augmentation(class2)

        f1 = plt.figure(1, figsize=(8, 8))
        # plt.subplot(2, 1, 1)
        # plt.ylim(-3, 5)
        # for signal in class1s:
        #     plt.plot(signal, c='c')
        # for signal in class1:
        #     plt.plot(t, signal, c='orange', label='92')
        # # plt.title('Normal in ECG5000')
        # plt.title('Normal in ECG200')
        # plt.legend()

        plt.subplot(2, 1, 1)
        plt.ylim(-3, 5)
        for signal in class1:
            plt.plot(t, signal, c='orange', label='92')
        # plt.title('Normal in ECG5000')
        plt.title('Normal in ECG200')
        plt.legend()
        #
        # plt.subplot(2, 2, 3)
        # plt.ylim(-3, 5)
        # for signal in class2s:
        #     plt.plot(signal, c='c')
        # plt.plot(t, class2[0], label='55')
        # plt.plot(t, class2[1], label='75')
        # plt.title('Ischemia in ECG200')
        # plt.legend()

        plt.subplot(2, 1, 2)
        plt.ylim(-3, 5)
        # for signal in class2:
        # plt.plot(t, class2[0], t, class2[1])
        plt.plot(t, class2[0])
        # plt.title('R-on-T PVC in ECG5000')
        plt.title('Ischemia in ECG200')
        plt.legend('55')

        # f2 = plt.figure(2)
        # plt.subplot(2, 1, 1)
        # plt.plot(t, class1_aug)
        # plt.title('class 1 in ECG5000 with white noise')
        # plt.subplot(2, 1, 2)
        # plt.plot(t, class2_aug)
        # plt.title('class 2 in ECG5000 with white noise')

    def plot_image(self):
        y, fea = self.load_data(self.root + self.dataset)
        print('feature shape: ', fea.shape)
        # idx1 = list(y).index(1)     # index of the first occurrence
        # idx2 = list(y).index(-1)

        idx1 = [i for i, label in enumerate(y) if label == 1]
        idx2 = [i for i, label in enumerate(y) if label == -1]
        # idx number in separate class set of the TEST set
        ms = random.sample(idx1, 5)
        ns = random.sample(idx2, 5)

        # print(fea[1,:].shape)     # (140,)
        class1s = fea[ms, :]    # normal
        class2s = fea[ns, :]    # abnormal
        print(class1s.shape)
        # repeat signal
        # class1 = np.concatenate((class1,class1,class1,class1))
        # class2 = np.concatenate((class2,class2,class2,class2))

        # to image
        im_rp1 = []
        im_rp2 = []
        im_gasf1 = []
        im_gasf2 = []
        im_mtf1 = []
        im_mtf2 = []

        for normal in class1s:
            normal = normal.reshape(1, -1)
            im_rp1.append(self.transform_to_2D('rp', normal))
            im_gasf1.append(self.transform_to_2D('gasf', normal))
            im_mtf1.append(self.transform_to_2D('mtf', normal))
        for abnormal in class2s:
            abnormal = abnormal.reshape(1, -1)
            im_rp2.append(self.transform_to_2D('rp', abnormal))
            im_gasf2.append(self.transform_to_2D('gasf', abnormal))
            im_mtf2.append(self.transform_to_2D('mtf', abnormal))

        # print()

        # print(im_rp1[0].shape)
        f1 = plt.figure(1)
        for i in range(0, len(im_rp1)):
            plt.subplot(2, len(im_rp1), i+1)
            plt.imshow(im_rp1[i][0], cmap='binary')
            plt.title('Normal_RP')
        for i in range(0, len(im_rp2)):
            plt.subplot(2, len(im_rp2), len(im_rp2)+i+1)
            plt.imshow(im_rp2[i][0], cmap='binary')
            plt.title('class2_RP')
        plt.suptitle('Comparison on ECG200')

        f2 = plt.figure(2)
        for i in range(0, len(im_gasf1)):
            plt.subplot(2, len(im_gasf1), i+1)
            plt.imshow(im_gasf1[i][0], cmap='binary')
            plt.title('Normal_GASF')
        for i in range(0, len(im_gasf2)):
            plt.subplot(2, len(im_gasf2), len(im_gasf2)+i+1)
            plt.imshow(im_gasf2[i][0], cmap='binary')
            plt.title('class2_GASF')
        plt.suptitle('Comparison on ECG200')

        f3 = plt.figure(3)
        for i in range(0, len(im_mtf1)):
            plt.subplot(2, len(im_mtf1), i+1)
            plt.imshow(im_mtf1[i][0], cmap='binary')
            plt.title('Normal_MTF')
        for i in range(0, len(im_mtf2)):
            plt.subplot(2, len(im_mtf2), len(im_mtf2)+i+1)
            plt.imshow(im_mtf2[i][0], cmap='binary')
            plt.title('class2_MTF')
        plt.suptitle('Comparison on ECG200')

        # f2 = plt.figure(2)
        # plt.subplot(2,1,1)
        # plt.imshow(im_gasf1[0], cmap='binary')
        # plt.title('Normal_GASF')
        # plt.subplot(2,1,2)
        # plt.imshow(im_gasf2[0], cmap='binary')
        # plt.title('class2_GASF')
        #
        # f3 = plt.figure(3)
        # plt.subplot(2,1,1)
        # plt.imshow(im_mtf1[0], cmap='binary')
        # plt.title('Normal_MTF')
        # plt.subplot(2,1,2)
        # plt.imshow(im_mtf2[0], cmap='binary')
        # plt.title('class2_MTF')
        return None


dataset = 'ECG200'
root = '../data/' + dataset + '/' + dataset + '/'
fname = dataset + '_TEST.txt'

drawing = Draw(root, fname)
# drawing.plot_image()
drawing.plot_signal()

# plt.tight_layout()
plt.show()


