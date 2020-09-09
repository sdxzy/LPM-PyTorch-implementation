import torch
from model import NetworkCIFAR as Network
import argparse
from genotypes import Genotype
from genotypes import PRIMITIVES
import torch.nn.functional as F
from torch.autograd import Variable
import utils
import torchvision.datasets as dset
#from operations import *
import pickle
import os
import time
from collections import OrderedDict
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#torch.manual_seed(0)
parser = argparse.ArgumentParser("sample")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--classes', type=int, default=10, help='total number of classes')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
args = parser.parse_args()

CIFAR_CLASSES = 10

class arch_generator():
    def __init__(self, step, multiper):
        self._steps = step
        self._multiplier = multiper
        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(torch.rand(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(torch.rand(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
    def genotype(self):

        def _parse(weights):
            k = sum(1 for i in range(self._steps) for n in range(2 + i))
            num_ops = len(PRIMITIVES)
            mask = torch.zeros(k, num_ops)
            #print(mask)
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                    mask[start+j, k_best] = 1
                start = end
                n += 1
            return gene, mask

        gene_normal, mask_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce, mask_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype, mask_normal, mask_reduce



def sample_net():

    arch = arch_generator(4, 4)
    arch._initialize_alphas()
    #print(arch.genotype())
    geno, mask_n, mask_r = arch.genotype()
    #print(mask_n)
    #print('-------------------------')
    #print(mask_r)
    model = Network(args.init_channels, args.classes, args.layers, False, geno)
    return model, mask_n, mask_r


def profile_once(test_queue, model):
    input, target = next(iter(test_queue))
    input = torch.unsqueeze(input[0, :, :, :], 0)
    input.cuda()
    with torch.no_grad():
        prof = profile_network(model, input, 1000, 1, True)
    print(1)
    return prof

def measure_latency_in_ms(model, input_shape, is_cuda):
    INIT_TIMES= 20
    LAT_TIMES = 30
    lat = AverageMeter()
    model.eval()

    x = torch.randn(input_shape)
    if is_cuda:
        model = model.cuda()
        x = x.cuda()
    else:
        model = model.cpu()
        x = x.cpu()

    with torch.no_grad():
        for _ in range(INIT_TIMES):
            output = model(x)

        for _ in range(LAT_TIMES):
            torch.cuda.synchronize()
            tic = time.time()
            output = model(x)
            torch.cuda.synchronize()
            toc = time.time()
            lat.update(toc - tic, x.size(0))

    return lat.avg * 1000   # save as ms

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def format_str(prof):
    for record in prof:
        cpu_time_avg = record[4]
        cuda_time_avg = record[5]
        params = record[6]
        input_shape = record[8]
        input_type = record[9]
        num_calls = record[10]
        op = record[11]


if __name__ == '__main__':
    list = OrderedDict()
    for i in range(100000):
        m, mask_n, mask_r = sample_net()
        input_shape = (64, 3, 32, 32)
        latency = measure_latency_in_ms(m, input_shape, True)
        mask_n = mask_n.view(-1)
        mask_r = mask_r.view(-1)
        #print(mask_n)
        mask = torch.cat([mask_n, mask_r], 0)
        mask = mask.numpy().astype(int)
        mask_str = str(mask.tolist())
        list[mask_str] = latency
        print('Iter {} \t      latency: {}'.format(i, latency))
        # if (i+1)%20000==0:
        #     with open('data_{}.pkl'.format(i), 'wb') as f:
        #         pickle.dump(list, f)
        #         f.close()
    with open('data.pkl', 'rb') as f:
        temp = pickle.load(f)
        f.close()

    #print(mask)
    #print(latency)
    #print(data)


