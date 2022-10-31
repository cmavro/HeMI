import torch.nn as nn
from layers import HGCN, AvgReadout, Discriminator
import torch

class DMGI(nn.Module):
    def __init__(self, nb_nodes, nfeat, nhid, shid, P, act, reg):
        super(DMGI, self).__init__()
        nhid = 256
        self.hgcn = HGCN(nfeat, nhid, shid, P, act)
        self.P =P
        self.H = nn.Parameter(torch.FloatTensor(1, nb_nodes, nhid))
        
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        #self.discs = nn.ModuleList()
        self.reg = reg
        self.init_weight()

        self.disc = Discriminator(nhid)
        
    def init_weight(self):
        nn.init.xavier_normal_(self.H)
        

    def forward(self, seq1, seq2, lbl, adjs, sparse, msk, samp_bias1, samp_bias2):
        b_xent = nn.BCEWithLogitsLoss()
        h_1, hh = self.hgcn(seq1, adjs, sparse)
        h_2, hh2 = self.hgcn(seq2, adjs, sparse)

        ret = 0
        for i in range(self.P):
            
            h1 = hh[i].unsqueeze(0)
            c = self.read(h1, msk)
            c = self.sigm(c)
            h2 = hh2[i].unsqueeze(0)
            logits = self.disc(c, h1, h2, samp_bias1, samp_bias2)
            ret += b_xent(logits,lbl)
        
        pos_reg_loss = ((self.H - h_1) ** 2).sum()
        neg_reg_loss = ((self.H - h_2) ** 2).sum()
        reg_loss = pos_reg_loss - neg_reg_loss
        

        return ret/self.P +  self.reg*reg_loss, hh

    # Detach the return variables
    def embed(self, seq, adjs, sparse, msk):
        h_1, hh = self.hgcn(seq, adjs, sparse)

        h =[]
        for i in range(self.P):
            h1 = hh[i].unsqueeze(0)
            h.append(h1.detach())
        #c = self.read(h_1, msk)

        #we have also tried self.H.detach(), self.H.detach() but works much worse
        return torch.cat(h, -1), h