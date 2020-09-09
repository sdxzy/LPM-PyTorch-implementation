import os
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from LPM_net import LPM
from MyDataset import MyDataset
import argparse
from torch.utils.tensorboard import SummaryWriter
import time
parser = argparse.ArgumentParser("LPM")
parser.add_argument('--epoch', type=int, default=1000, help='the number of training epoch')
parser.add_argument('--data_dir', type=str, default='./data/data_60000.pkl', help='dir of training data ')
parser.add_argument('--in_channel', type=int, default=224, help='the number of training epoch')
parser.add_argument('--learning_rate', type=float, default=0.005, help='the number of training epoch')
parser.add_argument('--momentum', type=float, default=0.9, help='the number of training epoch')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=256, help='the number of training epoch')
parser.add_argument('--save', type=str, default='./experiment', help='the number of training epoch')
parser.add_argument('--is_cuda', action='store_true', default=True, help='use gpu')
args = parser.parse_args()



save_dir = os.path.join(args.save, 'train-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
tb_logger_dir = save_dir + '/tb_logger'
if not os.path.exists(tb_logger_dir):
  os.makedirs(tb_logger_dir)
writer = SummaryWriter(tb_logger_dir)
model_dir = save_dir + '/weight'
if not os.path.exists(model_dir):
  os.makedirs(model_dir)


def main():

    ## data
    dataset = MyDataset(args.data_dir)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    ## model
    model = LPM(args.in_channel)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    ## loss
    criterion = nn.MSELoss()

    if args.is_cuda:
        model.cuda()
        criterion.cuda()

    ## train
    for i in range(args.epoch):
        tic= time.time()
        for stage, (arch, lat) in enumerate(dataloader):
            if args.is_cuda:
                arch = arch.cuda()
                lat = lat.cuda()
            sum_loss = 0
            stage_num = 0
            pre = model(arch)
            loss = criterion(pre, lat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss
            stage_num += 1
            if stage % 200 == 0:
                print('epoch: {}    stage: {}     loss:{}'.format(i, stage, loss))
        print('Epoch: {}    Avg_loss: {}     Time:{}'.format(i, sum_loss/stage_num, (time.time()-tic)))
        writer.add_scalar('train_loss', sum_loss/stage_num, i)

        if (i % 100==0 and i) or i ==(args.epoch-1):
            torch.save(model.state_dict(), '{}/model_{}.pt'.format(model_dir, i))


if __name__ == '__main__':
    main()