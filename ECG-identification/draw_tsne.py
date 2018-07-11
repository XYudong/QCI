import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd


def load_data(path):
    # load feature set
    # data = np.loadtxt(path, dtype=float)
    data = pd.read_csv(path, header=None)
    # print(type(data))
    # print(data.iloc[:, 0])
    y = data.iloc[:, 0]  # labels
    fea = data.iloc[:, 1:]
    return y, fea


def transform(fea, pca_n=40):
    pca = PCA(n_components=pca_n, random_state=666)
    fea_pca = pca.fit_transform(fea)
    tsne = TSNE(n_components=2, random_state=66, init='random')
    Y = tsne.fit_transform(fea_pca)
    return Y


def pca_n(data):
    # plot the curve of n_components vs variance ratio
    list_ratio = []
    for pca_n in range(0, 10):
        pca = PCA(n_components=pca_n, random_state=666)
        pca.fit(data)
        ratio_mat = pca.explained_variance_ratio_
        sum_ratio = np.sum(ratio_mat)
        list_ratio.append(sum_ratio)
    for pca_n in range(10, 61, 5):
        pca = PCA(n_components=pca_n, random_state=666)
        pca.fit(data)
        ratio_mat = pca.explained_variance_ratio_
        sum_ratio = np.sum(ratio_mat)
        list_ratio.append(sum_ratio)
        if pca_n==40:
            extra_tick = sum_ratio

    n = list(range(0,10))+list(range(10,61,5))

    f1 = plt.figure(1)
    plt.plot(n, list_ratio, linewidth=4)
    plt.plot([0,40], [extra_tick, extra_tick], 'r--')
    plt.plot([40,40], [0,extra_tick], 'r--')
    plt.title('PCA_ECG5000')
    plt.xlabel('n_components')
    plt.ylabel('ratio of variance')
    plt.xticks(list(range(0, 61, 5)))
    plt.yticks()
    plt.show()

# name of data set
fname = 'ECG200'
method = 'rp'

# y_train, fea_train = load_data(fname+'_'+method+'_fc1_features_train.txt')
# print(fea_train.shape)
y_test, fea_test = load_data(fname+'_'+method+'_fc1_features_test.csv')
# y_test, fea_test = load_data('temp.txt')
print(fea_test.shape)

# y_tr_te = np.concatenate((y_train, y_test))
# fea_tr_te = np.concatenate((fea_train, fea_test))
# print(fea_tr_te.shape)

pca_n = 40

# X_train = transform(fea_train, pca_n)
X_test = transform(fea_test, pca_n)
# X_tr_te = transform(fea_tr_te, pca_n)
# print('after transformation: ', X_tr_te.shape)
# bool_tr = y_train == 1
bool_te = y_test == 1
# bool_tr_te = y_tr_te == 1
# print(y_train)

# draw training and test data points separately
# f1 = plt.figure(1)
# for i in range(len(y_train)):
#     if y_train[i]==1:
#         class_1 = plt.scatter(X_train[i, 0], X_train[i, 1], c='dodgerblue')
#     elif y_train[i]==2:
#         class_2 = plt.scatter(X_train[i, 0], X_train[i, 1], c='slateblue')
#     elif y_train[i]==3:
#         class_3 = plt.scatter(X_train[i, 0], X_train[i, 1], c='tomato')
#     elif y_train[i]==4:
#         class_4 = plt.scatter(X_train[i, 0], X_train[i, 1], c='hotpink')
#     elif y_train[i]==5:
#         class_5 = plt.scatter(X_train[i, 0], X_train[i, 1], c='red')
#
# plt.title(fname+'_'+method+' training samples')
# plt.legend((class_1, class_2, class_3, class_4, class_5), ('class_1', 'class_2', 'class_3', 'class_4', 'class_5'))
#
# figname = fname+'_'+method+'_train_pca'+str(pca_n)
# plt.savefig(figname)

f2 = plt.figure(2)
for i in range(len(bool_te)):
    if bool_te[i]:
        test_good = plt.scatter(X_test[i, 0], X_test[i, 1], c='c', s=10)
    else:
        test_poor = plt.scatter(X_test[i, 0], X_test[i, 1], c='orange', s=10)
plt.title(fname+' test samples')
plt.legend((test_good, test_poor), ('test_good', 'test_poor'))

figname = fname+'_test_pca'+str(pca_n)
# plt.savefig(figname)

# # draw data points together
# f3 = plt.figure(3)
# for i in range(len(bool_te)):
#     if bool_te[i]:
#         test_good = plt.scatter(X_tr_te[i+len(bool_tr), 0], X_tr_te[i+len(bool_tr), 1], c='c', alpha=0.5, s=4)
#     else:
#         test_poor = plt.scatter(X_tr_te[i+len(bool_tr), 0], X_tr_te[i+len(bool_tr), 1], c='orange', alpha=0.5, s=4)
# for i in range(len(bool_tr)):
#     if bool_tr[i]:
#         train_good = plt.scatter(X_tr_te[i, 0], X_tr_te[i, 1], c='dodgerblue', s=20)
#     else:
#         train_poor = plt.scatter(X_tr_te[i, 0], X_tr_te[i, 1], c='r', s=20)
# plt.title(fname)
# plt.legend((train_good,train_poor,test_good,test_poor), ('train_good','train_poor','test_good','test_poor'))
#
# figname = fname+'_'+method+'_pca'+str(pca_n)
# plt.savefig(figname)

plt.show()


# pca_n(fea_tr_te)

