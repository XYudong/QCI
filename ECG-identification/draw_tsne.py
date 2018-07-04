import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_data(path):
    # load feature set
    data = np.loadtxt(path, dtype=float)
    y = data[:, 0]  # labels
    fea = data[:, 1:]
    return y, fea


def transform(fea, pca_n=50):
    pca = PCA(n_components=pca_n, random_state=666)
    fea_pca = pca.fit_transform(fea)
    tsne = TSNE(n_components=2, random_state=66, init='random')
    Y = tsne.fit_transform(fea_pca)
    return Y


# name of data set
fname = 'ECG5000'

y_train, fea_train = load_data(fname+'_fc1_features_train.txt')
print(fea_train.shape)
y_test, fea_test = load_data(fname+'_fc1_features_test.txt')

y_tr_te = np.concatenate((y_train, y_test))
fea_tr_te = np.concatenate((fea_train, fea_test))
print(fea_tr_te.shape)

pca_n = 50

# X_train = transform(fea_train, pca_n)
# X_test = transform(fea_test, pca_n)
X_tr_te = transform(fea_tr_te, pca_n)
print('after transformation: ', X_tr_te.shape)
bool_tr = y_train == 1
bool_te = y_test == 1
# bool_tr_te = y_tr_te == 1
# print(y_train)

# draw training and test data points separately
# f1 = plt.figure(1)
# for i in range(len(bool_tr)):
#     if bool_tr[i]:
#         train_good = plt.scatter(X_train[i, 0], X_train[i, 1], c='dodgerblue')
#     else:
#         train_poor = plt.scatter(X_train[i, 0], X_train[i, 1], c='r')
# plt.title(fname+' training samples')
# plt.legend((train_good, train_poor), ('train_good', 'train_poor'))
#
# figname = fname+'_train_pca'+str(pca_n)
# plt.savefig(figname)
#
# f2 = plt.figure(2)
# for i in range(len(bool_te)):
#     if bool_te[i]:
#         test_good = plt.scatter(X_test[i, 0], X_test[i, 1], c='c', s=5)
#     else:
#         test_poor = plt.scatter(X_test[i, 0], X_test[i, 1], c='orange', s=5)
# plt.title(fname+' test samples')
# plt.legend((test_good, test_poor), ('test_good', 'test_poor'))
#
# figname = fname+'_test_pca'+str(pca_n)
# plt.savefig(figname)

# draw data points together
f3 = plt.figure(3)
for i in range(len(bool_te)):
    if bool_te[i]:
        test_good = plt.scatter(X_tr_te[i+len(bool_tr), 0], X_tr_te[i+len(bool_tr), 1], c='c', alpha=0.5, s=4)
    else:
        test_poor = plt.scatter(X_tr_te[i+len(bool_tr), 0], X_tr_te[i+len(bool_tr), 1], c='orange', alpha=0.5, s=4)
for i in range(len(bool_tr)):
    if bool_tr[i]:
        train_good = plt.scatter(X_tr_te[i, 0], X_tr_te[i, 1], c='dodgerblue', s=20)
    else:
        train_poor = plt.scatter(X_tr_te[i, 0], X_tr_te[i, 1], c='r', s=20)
plt.title(fname)
plt.legend((train_good,train_poor,test_good,test_poor), ('train_good','train_poor','test_good','test_poor'))

figname = fname+'_pca'+str(pca_n)
plt.savefig(figname)
plt.show()

