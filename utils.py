import numpy as np
from sklearn.decomposition import PCA
from random import shuffle
import random
from sklearn import svm
import os
from torch.nn import init
from torch import nn
from scipy.io import loadmat


# 数据预处理
def preprocess(data, n_components=None):
    h, w, _ = data.shape
    if not data.dtype==np.float32:
        data = data.astype(np.float32)
    # 数据归一化
    data = data.reshape((h*w, -1))
    data -= np.min(data, axis=0)
    data /= np.max(data, axis=0)
    # PCA降维
    pca = PCA(0.99, whiten=True) if n_components is None else PCA(n_components)
    pca.fit(data)
    if len(pca.singular_values_) < 10:
        pca = PCA(10)
        data = pca.fit_transform(data)
    else:
        data = pca.transform(data)
    return data.reshape((h, w, -1))


# 挑选有价值的样本
def selectValuableSample(data, gt, iteration, n_select, seed=971104):
    '''
    :param data:[h, w, bands]
    :param gt: [h, w]
    :param iteration: 有价值样本挑选次数
    :param n_select: 每次挑选的样本数量
    :param seed: 随机种子
    :return:
    '''
    random.seed(971104)
    # 训练集每类初始样本量
    init_sample_size = 5
    # 训练集和候选池
    train_indices = []
    pool_indices = []
    for i in np.unique(gt):
        if i == 0:
            continue
        indices = list(zip(*np.nonzero(gt==i)))
        shuffle(indices)
        train_indices += indices[:init_sample_size]
        pool_indices += indices[init_sample_size:]
    train_indices = train_indices
    pool_indices = pool_indices
    net = svm.SVC(probability=True, random_state=seed)
    for i in range(iteration):
        X = data[tuple(zip(*train_indices))]
        Y = gt[tuple(zip(*train_indices))]
        net.fit(X, Y)
        X_hat = data[tuple(zip(*pool_indices))]
        prob = net.predict_proba(X_hat) # prob:[batch_size, nc]
        first_second = np.sort(prob, axis=-1)
        bvsb = first_second[:, -1] - first_second[:, -2]
        indices = np.argsort(bvsb)
        candidates = indices[:n_select]
        # 将下标索引从大到小进行排序，防止从候选池中剔除某样本后样本下标发生该表从而引发越界问题
        candidates.sort()
        candidates = candidates[::-1]
        # 将有价值的样本添加到训练数据集并从候选池中剔除
        for j in candidates:
            train_indices.append(pool_indices[j])
            pool_indices.pop(j)

    ans = np.zeros_like(gt, dtype=np.int)
    ans[tuple(zip(*train_indices))] = gt[tuple(zip(*train_indices))]
    return ans


def weight_init(m):
    if isinstance(m, nn.Linear):
        init.uniform_(m.weight, -1, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        init.uniform_(m.weight, -1, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def loadLabel(path):
    '''
    :param path:
    :return: 训练样本标签， 测试样本标签
    '''
    assert os.path.exists(path), '{},路径不存在'.format(path)
    # keys:{train_gt, test_gt}
    gt = loadmat(path)
    return gt['train_gt'], gt['test_gt']


from scipy.io import loadmat
m = loadmat('data/KSC/KSC.mat')
data = m['KSC']
# m = loadmat('trainTestSplit/Salinas/sample10_run0.mat')
# train_gt, test_gt = m['train_gt'], m['test_gt']
# data, train_gt, test_gt = data.astype(np.float32), train_gt.astype(np.int), test_gt.astype(np.int)
data = preprocess(data.astype(np.float32))
# ans = selectValuableSample(data, train_gt, 10, 3)
print(data.shape)


