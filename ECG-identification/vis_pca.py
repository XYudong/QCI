import numpy as np
from matplotlib import pyplot as plt

"""visualize PCA matrix in different classes with teh same color map """


def classify_mat(mat):
    label = mat[:, 0]
    idx_pos = [i for i, x in enumerate(label) if x == 1]
    idx_neg = [j for j, x in enumerate(label) if x == -1]
    mat_pos = mat[idx_pos, 1:]
    mat_neg = mat[idx_neg, 1:]

    return mat_pos, mat_neg


def vis_pca(dataset, name, n_comp):
    root = 'ECG200/'
    fname = dataset + '_comb_100'+name+'_pca'+str(n_comp)+'.npy'
    mat = np.load(root + fname)
    mat_pos, mat_neg = classify_mat(mat)
    n = len(mat_pos)
    print(n)
    mat_new = np.concatenate((mat_pos, mat_neg), axis=0)
    print(mat_new.shape)
    mat_new = np.repeat(mat_new, 10, axis=1)
    f1 = plt.matshow(mat_new, cmap=plt.cm.Blues, vmin=-20, vmax=30)
    xlim = f1.axes.get_xlim()
    plt.plot(xlim, [n, n], 'r')
    plt.colorbar()
    plt.title(name + ' set')
    plt.ylabel('samples')
    plt.xlabel('features\n after #%d is negative samples' % n)
    plt.xticks(np.arange(0,50,10), [1,2,3,4,5])


dataset = 'ECG200'
n_comp = 5
# vis_pca(dataset, 'test', n_comp)
vis_pca(dataset, 'train', n_comp)
plt.show()

