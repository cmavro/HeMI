import torch.nn as nn
import torch
from layers import HGCN, AvgReadout, Discriminator, GCN

class DGI(nn.Module):
    def __init__(self, nfeat, nhid, P, act):
        super(DGI, self).__init__()
        
        self.P = P
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.discs = nn.ModuleList()
        self.gcns = nn.ModuleList()
        for _ in range(self.P):
            self.gcns.append(GCN(nfeat, nhid, act))
            self.discs.append(Discriminator(nhid))

    def forward(self, seq1, seq2, lbl, adjs, sparse, msk, samp_bias1, samp_bias2):
        
        ret = 0
        hh =[]
        b_xent = nn.BCEWithLogitsLoss()
        for (i,adj) in enumerate(adjs):
            h_1 = self.gcns[i](seq1, adj, sparse)
            h_1 = h_1.unsqueeze(0)
            hh.append(h_1)
            c = self.read(h_1, msk)
        
            c = self.sigm(c)

            h_2 = self.gcns[i](seq2, adj, sparse)
            h_2 = h_2.unsqueeze(0)

            #print(c.size(), h_1.size(), h_2.size())
            logits = self.discs[i](c, h_1, h_2, samp_bias1, samp_bias2)
            ret += b_xent(logits,lbl)

        return ret/self.P, hh

    # Detach the return variables
    def embed(self, seq, adjs, sparse, msk):
        h =[]
        for (i,adj) in enumerate(adjs):
            h_1 = self.gcns[i](seq, adj, sparse)
            h_1 = h_1.unsqueeze(0)
            h.append(h_1.detach())
        #h_1 = self.gcn(seq, adjs, sparse)
        c = self.read(h_1, msk)

        return torch.cat(h, -1), h