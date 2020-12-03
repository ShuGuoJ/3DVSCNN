import torch
from torch.utils.data import Dataset
import numpy as np


class HSIDataset(Dataset):
    def __init__(self, data, gt, patchsz):
        super().__init__()
        self.indices = list(zip(*np.nonzero(gt)))
        self.gt = gt
        self.data = data
        self.h, self.w, self.bands = self.data.shape
        self.patchsz = patchsz
        # 添加镜像
        self.addMirror()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, y = self.indices[index]
        neighbor_region = self.data[x:x+self.patchsz, y:y+self.patchsz]
        label = self.gt[x, y] - 1
        return torch.tensor(neighbor_region, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


    # 添加镜像
    def addMirror(self):
        dx = self.patchsz // 2
        if dx != 0:
            mirror = np.zeros((self.h + 2 * dx, self.w + 2 * dx, self.bands))
            mirror[dx:-dx, dx:-dx, :] = self.data
            for i in range(dx):
                # 填充左上部分镜像
                mirror[:, i, :] = mirror[:, 2 * dx - i, :]
                mirror[i, :, :] = mirror[2 * dx - i, :, :]
                # 填充右下部分镜像
                mirror[:, -i - 1, :] = mirror[:, -(2 * dx - i) - 1, :]
                mirror[-i - 1, :, :] = mirror[-(2 * dx - i) - 1, :, :]
            self.data = mirror



class DatasetInfo(object):
    info = {'PaviaU': {
        'data_key': 'paviaU',
        'label_key': 'paviaU_gt',
        'n_component': 10,
        'patchsz': 13
    },
        'Salinas': {
            'data_key': 'salinas_corrected',
            'label_key': 'salinas_gt',
            'n_component': None,
            'patchsz': 13
        },
        'KSC': {
            'data_key': 'KSC',
            'label_key': 'KSC_gt',
            'n_component': 30,
            'patchsz': 13
    },  'Houston':{
            'data_key': 'Houston',
            'label_key': 'Houston2018_gt',
            'n_component': None,
            'patchsz': 13
    }}


# 验证数据集的准确性
# from scipy.io import loadmat
# m = loadmat('data/Salinas/Salinas.mat')
# data = m['salinas_corrected']
# m = loadmat('data/Salinas/Salinas_gt.mat')
# gt = m['salinas_gt']
# index = 1001
# indices = list(zip(*np.nonzero(gt)))
# x, y = indices[index]
# dataset = HSIDataset(data, gt, patchsz=13)
# neighbor_region, label = dataset[index]
# print(torch.equal(torch.tensor(data[x, y], dtype=torch.float), neighbor_region[6,6]))
# print(gt[x, y])
# print(label)

