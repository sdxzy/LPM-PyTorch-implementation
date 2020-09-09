import os
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from LPM_net import LPM
from MyDataset import MyDataset
import argparse
import time
import math
import matplotlib.pyplot as plt
from genotypes import Genotype,_parse
from genotypes import PRIMITIVES

class lat_loss():
    def __init__(self):
        self.loss = []


def test():

    is_cuda = True
    weight = './experiment/best/model_best.pt'

    loss_l = []
    loss_w_l = []
    noise = []

    ## data
    dataset = MyDataset('./data/data_test.pkl')
    dataloader = data.DataLoader(dataset,1,False)
    criterion = nn.MSELoss()

    ## model
    model = LPM(224)
    model.eval()
    if is_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    model.load_state_dict(torch.load(weight))

    ## test
    for stage, (arch, lat) in enumerate(dataloader):
        if is_cuda:
            arch = arch.cuda()
            lat = lat.cuda()
        predict = model(arch)

        ## loss
        loss = criterion(lat, predict).cpu().item()
        loss = math.sqrt(loss)
        label = lat.cpu().item()
        pre_lat = predict.cpu().item()
        print("Stage {:5d}:    Pre: {:8f}     Label: {:8f} ".format(stage, pre_lat, label))

        if abs(pre_lat-label) > 4:
            noise.append((arch, pre_lat, label))

        loss_l.append(loss)
        loss_w_l.append(loss/label)

    return loss_l, loss_w_l, noise


if __name__ == '__main__':
    loss_l, loss_w_l, noise = test()
    print('Loss mean:   {}'.format(np.mean(loss_l)))
    print('Loss std:   {}'.format(np.std(loss_l)))
    print('Loss AVG weight:  {}'.format(np.mean(loss_w_l)))
