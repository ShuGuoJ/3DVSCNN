import torch
from torch import nn, optim
import numpy as np
from scipy.io import loadmat
from utils import weight_init, loadLabel
from HSIDataset import HSIDataset, DatasetInfo
from Model.module import VSCNN_KSC, VSCNN_PaviaU
from torch.utils.data import DataLoader
import os
import argparse
from visdom import Visdom

isExists = lambda path: os.path.exists(path)
EPOCHS = 10
LR = 1e-1
BATCHSZ = 10
NUM_WORKERS = 8
SEED = 971104
torch.manual_seed(SEED)
DEVICE = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
viz = Visdom()


def train(model, criterion, optimizer, dataLoader):
    '''
    :param model: 模型
    :param criterion: 目标函数
    :param optimizer: 优化器
    :param dataLoader: 批数据集
    :return: 已训练的模型，训练损失的均值
    '''
    model.train()
    model.to(DEVICE)
    trainLoss = []
    for step, (neighbor_region, target) in enumerate(dataLoader):
        neighbor_region, target = neighbor_region.to(DEVICE), target.to(DEVICE)
        neighbor_region = neighbor_region.permute((0, 3, 1, 2)).unsqueeze(1)
        out = model(neighbor_region)

        loss = criterion(out, target)
        trainLoss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%5 == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('step:{} loss:{} lr:{}'.format(step, loss.item(), lr))
    return model, float(np.mean(trainLoss))


def test(model, criterion, dataLoader):
    model.eval()
    evalLoss, correct = [], 0
    for neighbor_region, target in dataLoader:
        neighbor_region, target = neighbor_region.to(DEVICE), \
                                  target.to(DEVICE)
        neighbor_region = neighbor_region.permute((0, 3, 1, 2)).unsqueeze(1)
        logits = model(neighbor_region)
        loss = criterion(logits, target)
        evalLoss.append(loss.item())
        pred = torch.argmax(logits, dim=-1)
        correct += torch.sum(torch.eq(pred, target).int()).item()
    acc = float(correct) / len(dataLoader.dataset)
    return acc, np.mean(evalLoss)


# def main(datasetName, n_sample_per_class, run, encoderPath=None):
#     # 加载数据和标签
#     info = DatasetInfo.info[datasetName]
#     data_path = "./data/{}/{}.mat".format(datasetName, datasetName)
#     label_path = './trainTestSplit/{}/sample{}_run{}.mat'.format(datasetName, n_sample_per_class, run)
#     isExists(data_path)
#     data = loadmat(data_path)[info['data_key']]
#     bands = data.shape[2]
#     isExists(label_path)
#     trainLabel, testLabel = loadLabel(label_path)
#     # 数据转换
#     data, trainLabel, testLabel = data.astype(np.float32), trainLabel.astype(np.int), testLabel.astype(np.int)
#     nc = int(np.max(trainLabel))
#     trainDataset = HSIDatasetV1(data, trainLabel, patchsz=42)
#     testDataset = HSIDatasetV1(data, testLabel, patchsz=42)
#     trainLoader = DataLoader(trainDataset, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)
#     testLoader = DataLoader(testDataset, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)
#     UNITS[0] = bands
#     model = SSDL(UNITS, nc)
#     model.apply(weight_init)
#     # 加载编码器的预训练参数
#     if encoderPath is not None:
#         isExists(encoderPath)
#         raw_dict = torch.load(encoderPath, map_location=torch.device('cpu'))
#         target_dict = model.encoder.state_dict()
#         keys = list(zip(raw_dict.keys(), target_dict.keys()))
#         for raw_key, target_key in keys:
#             target_dict[target_key] = raw_dict[raw_key]
#         model.encoder.load_state_dict(target_dict)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-2)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
#
#     for epoch in range(EPOCHS):
#         print('*'*5 + 'Epoch:{}'.format(epoch) + '*'*5)
#         model, trainLoss = train(model, criterion=criterion, optimizer=optimizer, dataLoader=trainLoader)
#         acc, evalLoss = test(model, criterion=criterion, dataLoader=testLoader)
#         viz.line([[trainLoss, evalLoss]], [epoch], win='train&test loss', update='append')
#         viz.line([acc], [epoch], win='accuracy', update='append')
#         print('epoch:{} trainLoss:{:.8f} evalLoss:{:.8f} acc:{:.4f}'.format(epoch, trainLoss, evalLoss, acc))
#         print('*'*18)
#         # if os.path.isdir('./ssdl/{}/ssdl_sample{}_run{}_epoch{}'.format(datasetName, n_sample_per_class, run, epoch))
#         if not os.path.isdir('./ssdl/{}'.format(datasetName)):
#             os.makedirs('./ssdl/{}'.format(datasetName))
#         torch.save(model.state_dict(), './ssdl/{}/ssdl_sample{}_run{}_epoch{}.pkl'.format(datasetName, n_sample_per_class, run, epoch))
#         scheduler.step()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='train ssdl')
#     parser.add_argument('--name', type=str, default='PaviaU',
#                         help='The name of dataset')
#     parser.add_argument('--epoch', type=int, default=1,
#                         help='模型的训练次数')
#     parser.add_argument('--lr', type=float, default=1e-1,
#                         help='learning rate')
#
#     args = parser.parse_args()
#     EPOCHS = args.epoch
#     datasetName = args.name
#     LR = args.lr
#     # main(datasetName, n_sample_per_class, run, encoderPath=None)
#     viz.line([[0., 0.]], [0.], win='train&test loss', opts=dict(title='train&test loss',
#                                                                 legend=['train_loss', 'test_loss']))
#     viz.line([0.,], [0.,], win='accuracy', opts=dict(title='accuracy',
#                                                      legend=['accuracy']))
#     main(datasetName, 10, 0, 'encoder/{}/{}_encoder_{}.ckpt'.format(datasetName, datasetName, 200))

