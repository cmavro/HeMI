
import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator, Discriminator_cluster, Clusterator
import torch.nn.functional as F
import numpy as np

class GIC(nn.Module):
    def __init__(self, n_nb, nfeat, nhid, P, act, num_clusters, beta, alpha):
        super(GIC, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.P = P
        
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        
        self.discs = nn.ModuleList()
        self.disc_cs = nn.ModuleList()
        self.gcns = nn.ModuleList()
        self.clusters = nn.ModuleList()
        for _ in range(self.P):
            self.gcns.append(GCN(nfeat, nhid, act))
            self.discs.append(Discriminator(nhid))
            self.disc_cs.append(Discriminator_cluster(nhid,nhid,n_nb,num_clusters))
            self.clusters.append(Clusterator(nhid,num_clusters))
        
        

    def forward(self, seq1, seq2, lbl, adjs, sparse, msk, samp_bias1, samp_bias2):
        ret = 0
        b_xent = nn.BCEWithLogitsLoss()
        hh = []
        for (i,adj) in enumerate(adjs):
            
            h_1 = self.gcns[i](seq1, adj, sparse)
            h_1 = h_1.unsqueeze(0)
            hh.append(h_1)
            h_2 = self.gcns[i](seq2, adj, sparse)
            h_2 = h_2.unsqueeze(0)
            
            

            Z, S = self.clusters[i](h_1[-1,:,:], self.beta)
            Z_t = S @ Z
            c2 = Z_t

            c2 = self.sigm(c2)

            c = self.read(h_1, msk)
            c = self.sigm(c) 
            
            

            logits =  self.alpha * self.discs[i](c, h_1, h_2, samp_bias1, samp_bias2)
            ret += self.alpha * b_xent(logits,lbl)
            logits = self.disc_cs[i](c2, c2,h_1[-1,:,:], h_1[-1,:,:] ,h_2[-1,:,:], S , samp_bias1, samp_bias2)
            ret += (1-self.alpha) * b_xent(logits,lbl)
        
        return ret/self.P, hh

    # Detach the return variables
    def embed(self, seq, adjs, sparse, msk):
        h = []
        for (i,adj) in enumerate(adjs):
            
            h_1 = self.gcns[i](seq, adj, sparse)
            h_1 = h_1.unsqueeze(0)
            h.append(h_1.detach())
        
        
        return torch.cat(h, -1), h