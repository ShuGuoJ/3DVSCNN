import torch
from torch import nn


class VSCNN_KSC(nn.Module):
    def __init__(self, bands, nc):
        # image_patch: 13x13
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv3d(1, 20, (9,3,3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),
            nn.Dropout3d(0.05),
            nn.Conv3d(20, 40, (9,3,3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear((bands-16)*40, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, nc)
        )

    def forward(self, input):
        '''
        :param input: [batchsz, 1, depth, h, w]
        :return: out: [batchsz, nc]
        '''
        f = self.feature(input)
        batchsz = f.shape[0]
        f = f.view((batchsz, -1))
        o = self.classifier(f)
        return o


class VSCNN_PaviaU(nn.Module):
    def __init__(self, bands, nc):
        # image_patch: 13x13
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv3d(1, 20, (3,3,3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),
            nn.Dropout3d(0.05),
            nn.Conv3d(20, 40, (3,3,3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear((bands-4)*40, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, nc)
        )

    def forward(self, input):
        '''
        :param input: [batchsz, 1, depth, h, w]
        :return: out: [batchsz, nc]
        '''
        f = self.feature(input)
        batchsz = f.shape[0]
        f = f.view((batchsz, -1))
        o = self.classifier(f)
        return o


# 测验Indian_pine数据集
# net = VSCNNi(30, 16)
# input = torch.rand((2, 1, 30, 13, 13))
# out = net(input)
# print(out.shape)

# 测验PaviaU数据集
# net = VSCNNp(10, 9)
# input = torch.rand((2, 1, 10, 13, 13))
# out = net(input)
# print(out.shape)
