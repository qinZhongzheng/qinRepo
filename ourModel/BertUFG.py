import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
import argparse
import sys
import logging
from datetime import datetime
from torch.optim import lr_scheduler
from model import BertClassifier,Net
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch.distributed as dist
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'])
parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nb_epochs', type=int, default=45)
parser.add_argument('-m', '--m', type=float, default=0.3, help='the factor balancing BERT and GCN prediction')
parser.add_argument('--ufg_lr', type=float, default=0.01,
                        help='learning rate (default: 5e-3)')
parser.add_argument('--wd', type=float, default=0.01,
                        help='weight decay (default: 5e-3)')
parser.add_argument('--nhid', type=int, default=16,
                        help='number of hidden units (default: 16)')
parser.add_argument('--Lev', type=int, default=2,
                        help='level of transform (default: 2)')
parser.add_argument('--s', type=float, default=2,
                        help='dilation scale > 1 (default: 2)')
parser.add_argument('--n', type=int, default=2,
                        help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
parser.add_argument('--FrameType', type=str, default='Haar',
                        help='frame type (default: Haar)')
parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability (default: 0.5)')
parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
parser.add_argument('--filename', type=str, default='results',
                        help='filename to store results and the model (default: results)')
parser.add_argument('--bert_lr', type=float, default=1e-5)
parser.add_argument('--bert_init', type=str, default='roberta-base',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--pretrained_bert_ckpt', default=None)
parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
args = parser.parse_args()
np.random.seed(args.seed)
th.manual_seed(args.seed)
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.nb_epochs
bert_init = args.bert_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
dropout = args.dropout
bert_lr = args.bert_lr

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, 'ufg', dataset)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

cpu = th.device('cpu')
# gpu = th.device('cpu')
# gpu = th.device('cuda:0')
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))

# Model

# function for pre-processing
@th.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = th.tensor(A.row)
    col = th.tensor(A.col)
    index = th.stack((row, col), dim=0)
    value = th.Tensor(A.data)

    return th.sparse_coo_tensor(index, value, A.shape)

# function for pre-processing
def ChebyshevApprox(f, n):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)

    return c

# function for pre-processing
def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]

    return d

# Data Preprocess
# 拿到TextGCN方法的邻接矩阵
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
'''
adj: n*n sparse adjacency matrix
y_train, y_val, y_test: n*c matrices 
train_mask, val_mask, test_mask: n-d bool array
'''

# compute number of real train/val/test/word nodes and number of classes
nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]
features_num = features.shape[1]

# Hyper-parameter Settings
learning_rate = args.ufg_lr
weight_decay = args.wd
nhid = args.nhid


# load documents and compute input encodings
corpse_file = './data/corpus/' + dataset +'_shuffle.txt'
with open(corpse_file, 'r') as f:
    text = f.read()
    text = text.replace('\\', '')
    text = text.split('\n')

def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    return input.input_ids, input.attention_mask

# transform one-hot label to class ID for pytorch computation
y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)

# document mask used for update feature
doc_mask  = train_mask + val_mask + test_mask

adj_norm = normalize_adj(adj + sp.eye(adj.shape[0])) #对adj做归一化
lobpcg_init = np.random.rand(nb_node, 1)
lambda_max, _ = lobpcg(adj_norm, lobpcg_init)
lambda_max = lambda_max[0]


# create index loader
train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
test_idx = Data.TensorDataset(th.arange(nb_node-nb_test, nb_node, dtype=th.long))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

# Training
def update_feature():
    global model, g, doc_mask
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    )
    with th.no_grad():
        model = model.to(device)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.to(device) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g

# extract decomposition/reconstruction Masks
#对邻接矩阵的操作
FrameType = args.FrameType

if FrameType == 'Haar':
    D1 = lambda x: np.cos(x / 2)
    D2 = lambda x: np.sin(x / 2)
    DFilters = [D1, D2]
    RFilters = [D1, D2]
elif FrameType == 'Linear':
    D1 = lambda x: np.square(np.cos(x / 2))
    D2 = lambda x: np.sin(x) / np.sqrt(2)
    D3 = lambda x: np.square(np.sin(x / 2))
    DFilters = [D1, D2, D3]
    RFilters = [D1, D2, D3]
elif FrameType == 'Quadratic':  # not accurate so far
    D1 = lambda x: np.cos(x / 2) ** 3
    D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
    D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
    D4 = lambda x: np.sin(x / 2) ** 3
    DFilters = [D1, D2, D3, D4]
    RFilters = [D1, D2, D3, D4]
else:
    raise Exception('Invalid FrameType')

Lev = args.Lev  # level of transform
s = args.s  # dilation scale
n = args.n  # n - 1 = Degree of Chebyshev Polynomial Approximation
J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1  # dilation level to start the decomposition
r = len(DFilters)

model = Net(nhid, nb_class, r, Lev, nb_node,m,shrinkage=None,
                    threshold=1e-3, dropout_prob=args.dropout).to(device)

# if th.cuda.device_count() > 1:
# model = nn.DataParallel(model)
# model = model.module

input_ids, attention_mask = encode_input(text, model.tokenizer)
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])

# build DGL Graph
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
g.ndata['label_train'] = th.LongTensor(y_train)
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))
logger.info('graph information:')
logger.info(str(g))


d = get_operator(adj, DFilters, n, s, J, Lev)
d_list = list()
for i in range(r):
    for l in range(Lev):
        d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))

if pretrained_bert_ckpt is not None:
    ckpt = th.load(pretrained_bert_ckpt, map_location=device)
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])


# optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer = th.optim.Adam([
        {'params': model.bert_model.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': bert_lr},
        {'params': model.GConv1.parameters(), 'lr': learning_rate},
        {'params': model.GConv2.parameters(), 'lr': learning_rate},
    ], lr=1e-3,weight_decay=weight_decay
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

def train_step(engine, batch):
    global model, g, optimizer
    model.train()
    model = model.to(device)
    g = g.to(device)
    optimizer.zero_grad()
    (idx, ) = [x.to(device) for x in batch]
    optimizer.zero_grad()
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    out = model(g,d_list,idx)
    y_pred = out[train_mask]
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc


trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()


def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(device)
        g = g.to(device)
        (idx, ) = [x.to(device) for x in batch]
        y_pred = model(g,d_list,idx)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true


evaluator = Engine(test_step)
metrics={
    'acc': Accuracy(),
    'nll': Loss(th.nn.NLLLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]
    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )
    if val_acc > log_training_results.best_val_acc :
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'ufg1': model.GConv1.state_dict(),
                'ufg2': model.GConv2.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc
        log_training_results.best_test_acc = test_acc
    if val_acc == log_training_results.best_val_acc and test_acc > log_training_results.best_test_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'ufg1': model.GConv1.state_dict(),
                'ufg2': model.GConv2.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_test_acc = test_acc


log_training_results.best_val_acc = 0
log_training_results.best_test_acc = 0
g = update_feature()
trainer.run(idx_loader, max_epochs=nb_epochs)

