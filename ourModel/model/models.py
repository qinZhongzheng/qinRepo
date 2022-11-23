import torch as th
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .torch_gcn import GCN
from .torch_gat import GAT


class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.classifier2 = th.nn.Linear(self.feat_dim,2)
        self.gcn = GCN( 
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        stauts_logit = self.classifier2(cls_feats)
        stauts_pred = th.nn.Softmax(dim=1)(stauts_logit)
        pred = th.log(pred)
        stauts_pred = th.log(stauts_pred)
        return pred,cls_feats,stauts_pred
 
class BertGAT(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, heads=8, n_hidden=32, dropout=0.5):
        super(BertGAT, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GAT(
                 num_layers=gcn_layers-1,
                 in_dim=self.feat_dim,
                 num_hidden=n_hidden,
                 num_classes=nb_class,
                 heads=[heads] * (gcn_layers-1) + [1],
                 activation=F.elu,
                 feat_drop=dropout,
                 attn_drop=dropout,
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g)[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred

class UFGConv(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, shrinkage=None, threshold=1e-4, bias=True):
        super(UFGConv, self).__init__()
        self.Lev = Lev
        self.shrinkage = shrinkage
        self.threshold = threshold
        self.crop_len = (Lev - 1) * num_nodes
        if th.cuda.is_available():
            self.weight = nn.Parameter(th.Tensor(in_features, out_features).cuda())
            self.filter = nn.Parameter(th.Tensor(r * Lev * num_nodes, 1).cuda())
        else:
            self.weight = nn.Parameter(th.Tensor(in_features, out_features))
            self.filter = nn.Parameter(th.Tensor(r * Lev * num_nodes, 1))
        if bias:
            if th.cuda.is_available():
                self.bias = nn.Parameter(th.Tensor(out_features).cuda())
            else:
                self.bias = nn.Parameter(th.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, d_list):
        # d_list is a list of matrix operators (torch sparse format), row-by-row
        # x is a torch dense tensor
        x = th.matmul(x, self.weight)

        # Fast Tight Frame Decomposition
        x = th.sparse.mm(th.cat(d_list, dim=0), x)
        # the output x has shape [r * Lev * num_nodes, #Features]

        # perform wavelet shrinkage (optional)
        if self.shrinkage is not None:
            if self.shrinkage == 'soft':
                x = th.mul(th.sign(x), (((th.abs(x) - self.threshold) + th.abs(th.abs(x) - self.threshold)) / 2))
            elif self.shrinkage == 'hard':
                x = th.mul(x, (th.abs(x) > self.threshold))
            else:
                raise Exception('Shrinkage type is invalid')

        # Hadamard product in spectral domain
        x = self.filter * x
        # filter has shape [r * Lev * num_nodes, 1]
        # the output x has shape [r * Lev * num_nodes, #Features]

        # Fast Tight Frame Reconstruction
        x = th.sparse.mm(th.cat(d_list[self.Lev - 1:], dim=1), x[self.crop_len:, :])

        if self.bias is not None:
            x += self.bias
        return x

# ufg的model与Bert联合预测的模型
class Net(nn.Module):
    def __init__(self,nhid, num_classes, r, Lev, num_nodes,m=0.7, shrinkage=None, threshold=1e-4, dropout_prob=0.5):
        super(Net, self).__init__()
        self.m = m
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.bert_model = AutoModel.from_pretrained('roberta-base')
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, num_classes)
        # self.GConv1 = UFGConv(self.feat_dim, num_classes, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold)
        self.GConv1 = UFGConv(self.feat_dim, nhid, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold)
        self.GConv2 = UFGConv(nhid, num_classes, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold)
        self.drop1 = nn.Dropout(dropout_prob)


    def forward(self, g, d_list,idx):
        # x = data.x  # x has shape [num_nodes, num_input_features] x为特征矩阵
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]

        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        # gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        # x = F.relu(self.GConv1(x, d_list))
        x = F.relu(self.GConv1(g.ndata['cls_feats'], d_list))
        # x = F.relu(self.GConv1(g.ndata['cls_feats'], d_list))[idx]
        x = self.drop1(x)
        x = self.GConv2(x, d_list)[idx]
        UFG_pred = th.nn.Softmax(dim=1)(x)
        pred = (UFG_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        # return F.log_softmax(x, dim=1)
        return pred
