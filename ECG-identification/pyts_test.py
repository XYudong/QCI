from pyts.image import MTF
from pyts.image import RecurrencePlots
import numpy as np
import matplotlib.pyplot as plt

# # Parameters
# n_samples, m_features = 100, 144
# # Toy dataset
# rng = np.random.RandomState(41)
# X = rng.randn(n_samples, m_features)
#
# MTF transfomation
# image_size = 48     # i.e (48, 48)
# mtf = MTF(image_size)
# X_mtf = mtf.fit_transform(X)
#
# # show
# plt.figure(1, figsize=(8,8))
# plt.imshow(X_mtf[0], cmap='rainbow', origin='lower')
#
#
# # Recurrence plot
# rp = RecurrencePlots(dimension=1,
#                      epsilon='percentage_points',
#                      percentage=30)
# X_rp = rp.fit_transform(X)
#
# plt.figure(2, figsize=(8,8))
# plt.imshow(X_rp[0],cmap='binary',origin='lower')
#
# plt.show()

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    #print(data[0:100])
    Y = data[:,0]
    X = data[:,1:]
    #print(X[0:100])
    return X, Y

def loaddataset():
    # root="D:\\dataset\\"
    root = "../data/"
    fname = 'ECG200'
    x_train, y_train = readucr(root+fname+'/'+fname+'/'+fname+'_TRAIN.txt')
    x_test, y_test = readucr(root+fname+'/'+fname+'/'+fname+'_TEST.txt')
    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test=loaddataset()
print(x_train.shape)

print("dimension before:", x_train.shape)
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))
print("dimension after:", x_train.shape)