
import torch
import torch.nn as nn
from layers import HGCN, AvgReadout, Discriminator, Discriminator_cluster, Clusterator
import torch.nn.functional as F
import numpy as np

class HGIC(nn.Module):
    def __init__(self, n_nb, nfeat, nhid, shid, P, act, num_clusters, beta, alpha):
        super(HGIC, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.P = P
        
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.hgcn = HGCN(nfeat, nhid, shid, P, act)
        self.disc = Discriminator(nhid) 
        self.disc_c = Discriminator_cluster(nhid,nhid,n_nb,num_clusters)
        self.cluster = Clusterator(nhid,num_clusters)
        
        
        

    def forward(self, seq1, seq2, lbl, adjs, sparse, msk, samp_bias1, samp_bias2):
        ret = 0
        b_xent = nn.BCEWithLogitsLoss()
        h_1, hh = self.hgcn(seq1, adjs, sparse)

        c = self.read(h_1, msk)
        
        c = self.sigm(c)

        h_2,_ = self.hgcn(seq2, adjs, sparse)
        
        Z, S = self.cluster(h_1[-1,:,:], self.beta)
        Z_t = S @ Z
        c2 = Z_t
        c2 = self.sigm(c2)

        logits = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        ret += self.alpha * b_xent(logits,lbl)
        logits = (1-self.alpha)*self.disc_c(c2, c2,h_1[-1,:,:], h_1[-1,:,:] ,h_2[-1,:,:], S , samp_bias1, samp_bias2)
        ret += (1-self.alpha)* b_xent(logits,lbl)

        
        return ret, hh

    # Detach the return variables
    def embed(self, seq, adjs, sparse, msk):
        h_all = []
        h_1,hh = self.hgcn(seq, adjs, sparse)
        for i in range(self.P):
            h1 = hh[i].unsqueeze(0)
            h_all.append(h1.detach())
        
        
        return h_1.detach(), h_all