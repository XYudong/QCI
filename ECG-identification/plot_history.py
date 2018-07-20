import numpy as np
from matplotlib import pyplot as plt
import pickle


def load_hist(fname):
    """load the dictionary of training history"""
    root = '../history/'
    with open(root + fname, 'rb') as infile:
        hist = pickle.load(infile, encoding='latin1')
    return hist


def plt_acc_loss(hist, fname):
    # summarize history for accuracy
    plt.figure(1, figsize=(8,10))
    plt.subplot(211)
    plt.plot(hist['acc'], c='dodgerblue', linewidth=2)
    plt.plot(hist['val_acc'], c='r')
    xlim = plt.gca().get_xlim()
    plt.plot(xlim, [0.9, 0.9], '--', c='seagreen')
    plt.ylim(0.3, 1.0)
    plt.grid(True)
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train acc', 'test acc '+str(format(max(hist['val_acc']), '.3f'))], loc='lower right')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(hist['loss'], c='dodgerblue')
    plt.plot(hist['val_loss'], c='r')
    plt.grid(True)
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'test loss'], loc='upper right')
    # figname = 'vgg16_ECG200_latest'
    # plt.savefig(fname)
    plt.tight_layout()

    plt.show()
    return None


fname = 'vgg16_ECG200_20'
history = load_hist(fname)

plt_acc_loss(history, fname)









