import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd


def load_data(path):
    # load feature set
    data = np.load(path)
    # data = pd.read_csv(path, header=None)
    # data = np.array(data)
    y = data[:, 0]    # labels-first
    fea = data[:, 1:]
    return y, fea


def do_pca(fea, n=20):
    # transform feature vector to 2D features
    print('applying PCA')
    pca = PCA(n_components=n, random_state=666)
    fea_pca = pca.fit_transform(fea)
    print(fea_pca.shape)
    return fea_pca


def do_tsne(fea_pca, name):
    print('\napplying TSNE')
    tsne = TSNE(n_components=2, random_state=66, init='random')
    fea_2d = tsne.fit_transform(fea_pca)
    print('save results of t-SNE')
    np.save('ECG200/' + name+'_tsne_2d', fea_2d)
    return fea_2d


def dump_data(dataset, method, train, test=np.array([])):
    root = 'ECG200/'
    if len(test):
        # df_tr = pd.DataFrame(train)
        # df_te = pd.DataFrame(test)
        # df_tr.to_csv(root + dataset + '_' + method + '_fla2D_' + 'train.csv', header=None, index=None)
        # df_te.to_csv(root + dataset + '_' + method + '_fla2D_' + 'test.csv', header=None, index=None)
        fname1 = dataset + '_' + method + '_100train_pca20.npy'
        fname2 = dataset + '_' + method + '_100test_pca20.npy'
        print('train: ', train.shape)
        print('test: ', test.shape)
        np.save(root + fname1, train)
        np.save(root + fname2, test)
    else:
        all_data = train
        df = pd.DataFrame(all_data)
        df.to_csv(root + dataset + '_' + method + '_fla2D.csv', header=None, index=None)
    print('data dumped')


def pca_n(data):
    # plot the curve of n_components vs variance ratio
    list_ratio = []
    for n_pca in range(0, 10):
        pca = PCA(n_components=n_pca, random_state=666)
        pca.fit(data)
        ratio_mat = pca.explained_variance_ratio_
        sum_ratio = np.sum(ratio_mat)
        list_ratio.append(sum_ratio)
    for n_pca in range(10, 50, 5):
        pca = PCA(n_components=n_pca, random_state=666)
        pca.fit(data)
        ratio_mat = pca.explained_variance_ratio_
        sum_ratio = np.sum(ratio_mat)
        list_ratio.append(sum_ratio)
        if n_pca == 15:
            extra_tick = sum_ratio

    n = list(range(0,10))+list(range(10,50,5))

    f1 = plt.figure(1)
    plt.plot(n, list_ratio, linewidth=4)
    plt.plot([0,15], [extra_tick, extra_tick], 'r--')
    plt.plot([15,15], [0,extra_tick], 'r--')
    plt.title('PCA_ECG5000')
    plt.xlabel('n_components')
    plt.ylabel('ratio of variance')
    # plt.xticks(list(range(0, 61, 5)))
    plt.yticks()
    plt.show()

    return True


def to_2d(fea_tr_te, y_tr_te, y_train, y_test):
    n_pca = 20
    print('transforming')
    # X_train = transform(fea_train, n_pca)
    # X_test = transform(fea_test, n_pca)
    X_tr_te = transform(fea_tr_te, n_pca)
    # print(X_tr_te.shape)

    # Xy_train = np.concatenate((X_train, y_train), axis=1)
    # Xy_test = np.concatenate((X_test, y_test), axis=1)
    Xy = np.concatenate((X_tr_te, y_tr_te), axis=1)
    print('Xy set: ', Xy.shape)

    dump_data(dataset, method, Xy)

    bool_tr = y_train == 1
    bool_te = y_test == 1
    # bool_tr_te = y_tr_te == 1
    # print(y_train)

    print('plotting')
    # draw training and test data points separately
    # f1 = plt.figure(1)
    # for i in range(len(y_train)):
    #     if y_train[i]==1:
    #         class_1 = plt.scatter(X_train[i, 0], X_train[i, 1], c='dodgerblue', s=4)
    #     elif y_train[i]==2:
    #         class_2 = plt.scatter(X_train[i, 0], X_train[i, 1], c='slateblue', s=4)
    #     # elif y_train[i]==3:
    #     #     class_3 = plt.scatter(X_train[i, 0], X_train[i, 1], c='tomato')
    #     # elif y_train[i]==4:
    #     #     class_4 = plt.scatter(X_train[i, 0], X_train[i, 1], c='hotpink')
    #     # elif y_train[i]==5:
    #     #     class_5 = plt.scatter(X_train[i, 0], X_train[i, 1], c='red')
    #
    # plt.title(dataset+'_'+method+'_2D training samples')
    # plt.legend((class_1, class_2), ('class_1', 'class_2'))
    #
    # figname = fname+'_'+method+'_train_pca'+str(n_pca)
    # plt.savefig(figname)

    # f2 = plt.figure(2)
    # for i in range(len(bool_te)):
    #     if bool_te[i]:
    #         test_good = plt.scatter(X_test[i, 0], X_test[i, 1], c='c', s=10)
    #     else:
    #         test_poor = plt.scatter(X_test[i, 0], X_test[i, 1], c='orange', s=10)
    # plt.title(fname+' test samples')
    # plt.legend((test_good, test_poor), ('test_good', 'test_poor'))
    #
    # figname = dataset+'_test_pca'+str(n_pca)
    # plt.savefig(figname)

    # draw data points together
    f3 = plt.figure(3)
    for i in range(len(bool_tr)):
        if bool_tr[i]:
            train_good = plt.scatter(X_tr_te[i, 0], X_tr_te[i, 1], c='c', alpha=0.8, s=10)
        else:
            train_poor = plt.scatter(X_tr_te[i, 0], X_tr_te[i, 1], c='orange', alpha=0.8, s=10)

    for i in range(len(bool_te)):
        if bool_te[i]:
            test_good = plt.scatter(X_tr_te[i+len(bool_tr), 0], X_tr_te[i+len(bool_tr), 1], c='dodgerblue', s=20)
        else:
            test_poor = plt.scatter(X_tr_te[i+len(bool_tr), 0], X_tr_te[i+len(bool_tr), 1], c='r', s=20)
    # fp = plt.scatter(X_tr_te[[9+len(bool_tr)], 0], X_tr_te[[9+len(bool_tr)], 1],
    #                  c='purple', s=25)
    plt.title(dataset+'_'+method+'_flatten_2D')
    # plt.legend((train_good,train_poor,test_good,test_poor, fp), ('train_Normal','train_Ischemia',
    #                                                              'test_Normal','test_Ischemia', 'false_positive'))
    #
    plt.legend((train_good, train_poor, test_good, test_poor), ('train_class1', 'train_class2', 'test_class1', 'test_class2'))

    # figname = fname+'_'+method+'_pca'+str(n_pca)
    # plt.savefig(figname)

    plt.show()
    return None


# def draw():


# name of data set
dataset = 'ECG200'
method = 'comb'
path = 'ECG200/'
print('loading data')
y_train, fea_train = load_data(path + dataset+'_'+method+'_100train_vgg.npy')
y_test, fea_test = load_data(path + dataset+'_'+method+'_100test_vgg.npy')
# y_test, fea_test = load_data('temp.txt')
print('training set: ', fea_train.shape)
print('test set: ', fea_test.shape)
# print(y_train.shape)      # (140,)

# fea_tr_te = np.concatenate((fea_train, fea_test))
# y_tr_te = np.concatenate((y_train, y_test))
# print(fea_tr_te.shape)


# y_tr_te = np.expand_dims(y_tr_te, axis=1)
# print(y_train.shape)      # (140, 1)

# to_2d(fea_tr_te, y_tr_te, y_train, y_test)


fea_train_new = do_pca(fea_train, 20)
fea_test_new = do_pca(fea_test, 20)

fea_all_new = np.concatenate((fea_train_new, fea_test_new))

# y_train = np.expand_dims(y_train, axis=1)
# y_test = np.expand_dims(y_test, axis=1)
# Xy_train = np.concatenate((y_train, fea_train_new), axis=1)
# Xy_test = np.concatenate((y_test, fea_test_new), axis=1)

# dump_data(dataset, method, Xy_train, Xy_test)
do_tsne(fea_all_new, 'ALL')       # only call this one time


# pca_n(fea_test)

