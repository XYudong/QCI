import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm


def load_data(path):
    # load 2D feature set
    data = pd.read_csv(path, header=None)
    # print(type(data))
    # print(data.iloc[:, 0])
    y = np.array(data.iloc[:, -1])   # labels-last
    fea_2D = np.array(data.iloc[:, 0:-1])
    return y, fea_2D


def preprocess_data(X_2D_tr, X_2D_te, y_tr, y_te, dataset):
    if dataset == 'ECG200':
        y = np.concatenate((y_tr, y_te))
        X_2D = np.concatenate((X_2D_tr, X_2D_te))
        print('X shape: ', X_2D.shape)
        X_train, X_test, y_train, y_test = train_test_split(X_2D, y, test_size=0.25, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X_2D_tr, X_2D_te, y_tr, y_te
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, dataset):
    print('start training')
    if dataset == 'ECG200':
        C = 5
        gamma = 2
    elif dataset == 'ECG5000':
        C = 1
        gamma = 0.1
    else:
        print('invalid dataset name')
        return None
    clf = svm.SVC(C=C, gamma=gamma, random_state=66)
    clf.fit(X_train, y_train)
    print('start testing')
    mean_acc = clf.score(X_test, y_test)
    print('Mean accuracy: ', mean_acc)
    return clf, mean_acc


def plot_data(groups, dataset, acc):
    if dataset == 'ECG200':
        for l, group in groups:
            if l == -1:
                l = 'Ischemia'
            else:
                l = 'Normal'
            plt.plot(group.x, group.y, 'o', label=l, ms=5)
            plt.title(dataset + '_train_2D' + '  accuracy: ' + str(acc))
        plt.legend()
    elif dataset == 'ECG5000':
        for l, group in groups:
            if l == 1:
                l = 'class 1'
            elif l == 2:
                l = 'class 2'
            plt.plot(group.x, group.y, 'o', label=l, ms=3)
            plt.title(dataset + '_train_2D' + '  accuracy: ' + str(acc))
        plt.legend()
    else:
        print('invalid dataset name')
        return False


def plot_boundary(model):
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, cmap='Paired')
    return None


def dump_model(dataset, model):
    fileout = open('svm_' + dataset + '_model', 'w+b')
    pickle.dump(model, fileout)
    fileout.close()
    print('model saved')
    return None


dataset = 'ECG200'
fname1 = dataset + '_comb_2D_train.csv'
fname2 = dataset + '_comb_2D_test.csv'
y_tr, X_2D_tr = load_data(fname1)
y_te, X_2D_te = load_data(fname2)

X_train, X_test, y_train, y_test = preprocess_data(X_2D_tr, X_2D_te, y_tr, y_te, dataset)

clf, acc = train_model(X_train, X_test, y_train, y_test, dataset)
# dump_model(dataset, clf)
# # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap='rainbow')

# plot data points
df = pd.DataFrame(dict(x=X_train[:, 0], y=X_train[:, 1], label=y_train))
groups = df.groupby('label')

f1 = plt.figure(1)
plot_data(groups, dataset, acc)

# plot decision boundary
# plot_boundary(clf)

plt.show()


