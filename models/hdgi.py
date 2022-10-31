import torch.nn as nn
from layers import HGCN, AvgReadout, Discriminator

class HDGI(nn.Module):
    def __init__(self, nfeat, nhid, shid, P, act):
        super(HDGI, self).__init__()
        self.hgcn = HGCN(nfeat, nhid, shid, P, act)
        self.P = P
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(nhid)

    def forward(self, seq1, seq2, lbl, adjs, sparse, msk, samp_bias1, samp_bias2):
        b_xent = nn.BCEWithLogitsLoss()

        h_1, hh = self.hgcn(seq1, adjs, sparse)

        c = self.read(h_1, msk)
        
        c = self.sigm(c)

        h_2,_ = self.hgcn(seq2, adjs, sparse)

        logits = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        ret = b_xent(logits,lbl)
        return ret, hh

    # Detach the return variables
    def embed(self, seq, adjs, sparse, msk):
        h_all = []
        h_1, hh = self.hgcn(seq, adjs, sparse)
        c = self.read(h_1, msk)
        for i in range(self.P):
            h1 = hh[i].unsqueeze(0)
            h_all.append(h1.detach())
        #     print(h1.size())
        # print(h_all)
        return h_1.detach(), h_all