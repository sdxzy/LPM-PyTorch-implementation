from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)


def _parse(weights, step):
  gene = []
  n = 2
  start = 0
  for i in range(step):
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
    start = end
    n += 1
  return gene



DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
DARTS_Second_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

DARTS_Second_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 3), ('sep_conv_5x5', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

DARTS_first = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 3), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

DARTS_First_20 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

#DARTS_Second_2 = DARTS_V2

