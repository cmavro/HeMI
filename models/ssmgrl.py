
import torch
import torch.nn as nn
from layers import HGCN
import torch.nn.functional as F
import numpy as np
EPS = 1e-8
class SSMGRL(nn.Module):
    def __init__(self, nfeat, nhid, shid, P, act):
        super(SSMGRL, self).__init__()
        self.P = P
        self.nhid = nhid
        self.eye = torch.eye(nhid,nhid).cuda()

        self.hgcn = HGCN(nfeat, nhid, shid, P, act)
        self.mlps = nn.ModuleList()
        for i in range(P):
            self.mlps.append(nn.Linear(nfeat, nhid))

    def forward(self, seq1, seq2, lbl, adjs, sparse, msk, samp_bias1, samp_bias2):
        ret = 0
        xs = []
        b_xent = nn.BCEWithLogitsLoss()
        h_1, hh = self.hgcn(seq1, adjs, sparse)

        for i in range(self.P):
            xs.append(F.relu(self.mlps[i](seq1))[0,:])

        for i in range(self.P):
            x = xs[i] / (xs[i].norm(dim=1)[:, None] + EPS)
            h = hh[i] / (hh[i].norm(dim=1)[:, None]+ EPS)
            #print(x.size(), h.size())
            ret += torch.norm(x.t() @ h - self.eye)

        for i in range(self.P):
            h1 = hh[i] / (hh[i].norm(dim=1)[:, None]+ EPS)
            for j in range(self.P):
                if j == i : continue
                h2 = hh[j] / (hh[j].norm(dim=1)[:, None]+ EPS)
                ret += torch.norm(h1.t() @ h2 - self.eye)
        
        
        return ret, hh

    # Detach the return variables
    def embed(self, seq, adjs, sparse, msk):
        h = []
        h_1,hh = self.hgcn(seq, adjs, sparse)
        for i in range(self.P):
            h1 = hh[i] / (hh[i].norm(dim=1)[:, None]+ EPS)
            h1 = h1.unsqueeze(0)
            h.append(h1.detach())
        
        
        return torch.cat(h,-1), h