import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm


def load_data(path):
    # load 2D feature set
    # data = pd.read_csv(path, header=None)
    # data = np.array(data)
    data = np.load(path)
    y = data[:, 0]
    fea_2D = data[:, 1:]
    return y, fea_2D


def train_model(X_train, y_train, dataset):
    print('start training')
    if dataset == 'ECG200':
        C = 5
        gamma = 0.01
    elif dataset == 'ECG5000':
        C = 5
        gamma = 0.1
    else:
        print('invalid dataset name')
        return None
    clf = svm.SVC(kernel='linear', C=C, gamma=gamma, random_state=66)
    clf.fit(X_train, y_train)
    return clf


def get_score(clff, x, y, name='test'):
    acc = clff.score(x, y)
    print(name + ' accuracy: ', acc)
    return acc


def prediction(clf, X, y):
    preds = clf.predict(X)
    diff = y - preds
    FN = []
    FP = []
    TP = []
    TN = []
    for i in range(len(diff)):
        if diff[i] == 0:
            if y[i] == 1:
                TP.append(i)
            elif y[i] == -1:
                TN.append(i)
            else:
                print('invalid labels')
                return None
        elif diff[i] == -2:
            FP.append(i)
        elif diff[i] == 2:
            FN.append(i)
        else:
            print('invalid labels or predictions')
            return None
    return TP, TN, FP, FN, diff


def revise_labels(diffs, TP, TN):
    diffs[TP] = 0.1
    diffs[TN] = -0.1
    return diffs


def plot_data(groups, dataset, name, acc=[], ms=5, shape='o'):
    if dataset == 'ECG200':
        for tag, group in groups:
            if tag == 0.1:
                tag = 'TP'
            elif tag == -0.1:
                tag = 'TN'
            elif tag == -2:
                tag = 'FP'
            elif tag == 2:
                tag = 'FN'
            plt.plot(group.x, group.y, shape, label=tag, ms=ms)
            plt.xlabel('o for TRAIN, ^ for TEST')
            if acc:
                plt.title(dataset + '_SVM_' + name+' TEST_acc: ' + str(acc))
            else:
                plt.title(dataset + '_SVM_' + name)
        plt.legend()
    elif dataset == 'ECG5000':
        for tag, group in groups:
            if tag == 1:
                tag = 'class 1'
            elif tag == 2:
                tag = 'class 2'
            plt.plot(group.x, group.y, shape, label=tag, ms=ms)
            plt.title(dataset + '_SVM_' + name+'_accuracy: ' + str(acc))
        plt.legend()
    else:
        print('invalid dataset name')
        return False


def plot_boundary(figure, model):
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    # print('XX shape: ', XX.shape)     # (30, 30)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy)
    # print('Z shape: ', Z.shape)   # (900,)
    Z = Z.reshape(XX.shape)

    # plot decision boundary and margins
    cax = ax.contourf(XX, YY, Z, cmap='Paired')
    figure.colorbar(cax)
    return None


def dump_model(dataset, model):
    # root = '../weights/'
    root = 'ECG200/SVM/'
    with open(root + 'svm_' + dataset + '_model', 'w+b') as outfile:
        pickle.dump(model, outfile)
    print('SVM model saved')
    return None


root = 'ECG200/'
dataset = 'ECG200'
method = 'comb'
fname1 = dataset + '_' + method + '_100train_pca5.npy'
fname2 = dataset + '_' + method + '_100test_pca5.npy'
y_tr, X_tr = load_data(root + fname1)
y_te, X_te = load_data(root + fname2)

# fname = root + dataset + '_comb_2D.csv'
# y, X_2D = load_data(fname)

# X_2D_tr, X_2D_te, y_tr, y_te = train_test_split(X_2D, y, test_size=0.25, random_state=55)


clf = train_model(X_tr, y_tr, dataset)
dump_model(dataset, clf)
train_acc = get_score(clf, X_tr, y_tr, 'Training')
test_acc = get_score(clf, X_te, y_te, 'Test')


# Analysis
TP_te, TN_te, FP_te, FN_te, diff_te = prediction(clf, X_te, y_te)
print('FP: ', FP_te)
print('FN: ', FN_te)

TP_tr, TN_tr, FP_tr, FN_tr, diff_tr = prediction(clf, X_tr, y_tr)

# improve diffs of labels
diff_tr = revise_labels(diff_tr, TP_tr, TN_tr)
diff_te = revise_labels(diff_te, TP_te, TN_te)

# load t-SNE coordinates of TEST set
# coord_te = np.load(root + 'TEST_tsne_2d.npy')      # (100, 2)
# coord_tr = np.load(root + 'TRAIN_tsne_2d.npy')
coord_all = np.load(root + 'ALL_tsne_2d.npy')

# # plot TEST data points
# df = pd.DataFrame(dict(x=coord_te[:, 0], y=coord_te[:, 1], label=diff_te))
# groups_te = df.groupby('label')
#
# # plot TRAIN data points
# df_tr = pd.DataFrame(dict(x=coord_tr[:, 0], y=coord_tr[:, 1], label=diff_tr))
# groups_tr = df_tr.groupby('label')

# plot ALL data points
df_all_1 = pd.DataFrame(dict(x=coord_all[0:100, 0], y=coord_all[0:100, 1], label=diff_tr))
df_all_2 = pd.DataFrame(dict(x=coord_all[100:, 0], y=coord_all[100:, 1], label=diff_te))
groups_all_1 = df_all_1.groupby('label')
groups_all_2 = df_all_2.groupby('label')

# f1 = plt.figure(1)
# plot_data(groups_tr, dataset, 'TRAIN', train_acc)
# f2 = plt.figure(2)
# plot_data(groups_te, dataset, 'TEST', test_acc)
f3 = plt.figure(3)
plot_data(groups_all_1, dataset, 'ALL', ms=4, shape='o')       # TRAIN set
plot_data(groups_all_2, dataset, 'ALL', ms=6, shape='^', acc=test_acc)       # TEST set

plt.show()


