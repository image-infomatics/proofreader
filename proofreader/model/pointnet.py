from __future__ import print_function
from os import fwalk
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, num_points, batch_norm=True):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        if batch_norm:
            bn = nn.BatchNorm1d
        else:
            bn = nn.Identity

        self.bn1 = bn(64)
        self.bn2 = bn(128)
        self.bn3 = bn(1024)
        self.bn4 = bn(512)
        self.bn5 = bn(256)

        self.register_buffer('iden', torch.tensor(
            [1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32))

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x[:] + self.iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points, global_feat=True, batch_norm=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points, batch_norm=batch_norm)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        print(x.shape, trans.shape)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans


class PointNet(nn.Module):
    def __init__(self, num_points, classes=2, batch_norm=True):
        super(PointNet, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(
            num_points, global_feat=True, batch_norm=batch_norm)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=-1)
        return x


if __name__ == '__main__':
    torch.manual_seed(7)
    num_points = 1024
    batch_size = 1
    num_feature = 3
    classes = 2
    print(
        f'num_points {num_points}, batch_size {batch_size}, num_feature {num_feature} classes {classes}')

    sim_data = torch.rand(batch_size, num_feature, num_points)

    # trans = STN3d(num_points=num_points)
    # out = trans(sim_data)
    # print('stn', out.size())
    # print('stn', out)

    # pointfeat = PointNetfeat(num_points=num_points, global_feat=True)
    # out, _ = pointfeat(sim_data)
    # print('global feat', out.size())

    # pointfeat = PointNetfeat(num_points=num_points, global_feat=False)
    # out, _ = pointfeat(sim_data)
    # print('point feat', out.size())

    model = PointNet(num_points=num_points, classes=classes, batch_norm=False)
    model.eval()
    y_hat = model(sim_data)
    print('class', y_hat.size())
    print(y_hat)
    threshold = 0.5

    pred_class = (y_hat > threshold).int()

    print('pred_class', pred_class)
