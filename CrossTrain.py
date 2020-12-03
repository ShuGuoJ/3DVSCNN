import torch
from torch import nn, optim
import numpy as np
from scipy.io import loadmat, savemat
from utils import weight_init, loadLabel, preprocess, selectValuableSample
from HSIDataset import HSIDataset, DatasetInfo
from Model.module import VSCNN_KSC, VSCNN_PaviaU
from torch.utils.data import DataLoader
import os
import argparse
from visdom import Visdom
from train import train, test


isExists = lambda path: os.path.exists(path)
SAMPLE_PER_CLASS = [10, 50, 100]
RUN = 10
EPOCHS = 10
LR = 1e-1
BATCHSZ = 10
NUM_WORKERS = 8
SEED = 971104
torch.manual_seed(SEED)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
viz = Visdom()
ROOT = None
N_SELECT = 4

def main(datasetName, n_sample_per_class, run):
    # 加载数据和标签
    info = DatasetInfo.info[datasetName]
    data_path = "./data/{}/{}.mat".format(datasetName, datasetName)
    label_path = './trainTestSplit/{}/sample{}_run{}.mat'.format(datasetName, n_sample_per_class, run)
    isExists(data_path)
    data = loadmat(data_path)[info['data_key']]

    isExists(label_path)
    trainLabel, testLabel = loadLabel(label_path)
    res = torch.zeros((3, EPOCHS))
    # 数据转换
    data, trainLabel, testLabel = data.astype(np.float32), trainLabel.astype(np.int), testLabel.astype(np.int)
    # 数据预处理
    data = preprocess(data, info['n_component'])
    bands = data.shape[2]
    # 挑选有价值的样本
    s = int(np.sum(trainLabel != 0))
    iteration = int(np.math.ceil((0.8 * s - 90) / N_SELECT))
    trainLabel = selectValuableSample(data, trainLabel, iteration, N_SELECT)

    nc = int(np.max(trainLabel))
    trainDataset = HSIDataset(data, trainLabel, patchsz=info['patchsz'])
    testDataset = HSIDataset(data, testLabel, patchsz=info['patchsz'])
    trainLoader = DataLoader(trainDataset, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)
    testLoader = DataLoader(testDataset, batch_size=128, shuffle=True, num_workers=NUM_WORKERS)

    model = VSCNN_KSC(bands, nc) if datasetName == 'KSC' else VSCNN_PaviaU(bands, nc)
    # model.apply(weight_init)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    for epoch in range(EPOCHS):
        print('*'*5 + 'Epoch:{}'.format(epoch) + '*'*5)
        model, trainLoss = train(model, criterion=criterion, optimizer=optimizer, dataLoader=trainLoader)
        acc, evalLoss = test(model, criterion=criterion, dataLoader=testLoader)
        print('epoch:{} trainLoss:{:.8f} evalLoss:{:.8f} acc:{:.4f}'.format(epoch, trainLoss, evalLoss, acc))
        print('*'*18)
        res[0][epoch], res[1][epoch], res[2][epoch] = trainLoss, evalLoss, acc
        if epoch%5==0:
            torch.save(model.state_dict(), os.path.join(ROOT, 'vscnn_sample{}_run{}_epoch{}.pkl'.format(n_sample_per_class,
                                                                                                       run, epoch)))
        scheduler.step()
    tmp = res.numpy()
    savemat(os.path.join(ROOT, 'res.mat'), {'trainLoss':tmp[0], 'evalLoss':tmp[1], 'acc':tmp[2]})
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train VSCNN')
    parser.add_argument('--name', type=str, default='Salinas',
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=1,
                        help='模型的训练次数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')

    args = parser.parse_args()
    EPOCHS = args.epoch
    datasetName = args.name
    LR = args.lr
    print('*'*5 + datasetName + '*'*5)
    for i, num in enumerate(SAMPLE_PER_CLASS):
        print('*' * 5 + 'SAMPLE_PER_CLASS:{}'.format(num) + '*' * 5)
        res = torch.zeros((RUN, 3, EPOCHS))
        for r in range(RUN):
            print('*' * 5 + 'run:{}'.format(r) + '*' * 5)
            ROOT = 'vscnn/{}/{}/{}'.format(datasetName, num, r)
            if not os.path.isdir(ROOT):
                os.makedirs(ROOT)
            res[r] = main(datasetName, num, r)
        mean = torch.mean(res, dim=0)  # [3, EPOCHS]
        viz.line(mean.T, list(range(EPOCHS)), win='SAMPLE_PER_CLASS_{}'.format(num), opts=dict(title='SAMPLE_PER_CLASS_{}'.format(num),
                                                                                               legend=['train loss', 'test loss', 'acc']))
    print('*'*5 + 'FINISH' + '*'*5)