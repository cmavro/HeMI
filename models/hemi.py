import torch.nn as nn
from layers import HGCN, HGCN_shared, AvgReadout, Discriminator, Discriminator2
import torch
import torch.nn.functional as fn

class HEMI(nn.Module):
    def __init__(self, n_nb, nfeat, nhid, shid, P, act, lam, dataset, hards = True):
        super(HEMI, self).__init__()
        self.dataset = dataset
        self.hards = hards
        self.P = P
        self.lam = lam 
        

        self.hgcns = nn.ModuleList() 
        self.hgcns.append(HGCN(nfeat, nhid, shid, P, act))
        if hards:
            self.hgcns.append(HGCN_shared(nfeat, nhid, shid, P, act))
        
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        
        self.discs = nn.ModuleList()
        self.discs0 = nn.ModuleList()
        self.discs2 = nn.ModuleList()

        for _ in range(P):
            self.discs.append(Discriminator(nhid))
            self.discs0.append(Discriminator(nhid))
            self.discs2.append(Discriminator2(nhid))

        
    def forward(self, seq1, seq2, lbl, adjs, sparse, msk, samp_bias1, samp_bias2):
        
        b_xent = nn.BCEWithLogitsLoss()
        
        ret = 0
        ret2 = 0
        #_, xm0 = self.hgcns[0](seq1, adjs, sparse)

        for hgcn in self.hgcns:
            x, xm = hgcn(seq1, adjs, sparse)
            x2, xm2 = hgcn(seq2, adjs, sparse)
            if self.lam!=0:
                for i in range(len(xm)):
                    c1 = xm[i].unsqueeze(0)   #use this for Acm, imdb node classification/clustering
                    c1 = self.sigm(c1)
                    logits = self.discs2[i](c1, x, x2, samp_bias1, samp_bias2)
                    ret += b_xent(logits,lbl)
                    
            if self.lam!=1:
                for i in range(len(xm)):
                    #c = self.read(xm[i],msk)
                    c2 = self.sigm(self.read(xm[i],msk))
                    logits = self.discs[i](c2, x, x2, samp_bias1, samp_bias2)
                    ret2 += b_xent(logits,lbl)
                    

        ret= ret*self.lam*(1/self.P)
        ret2= ret2*(1-self.lam)*(1/self.P)

        return ret+ret2, xm


        
    # Detach the return variables
    def embed(self, seq, adjs, sparse, msk):
        h =[]
        h_all = []
        #for hgcn in self.hgcns:
        x, hh = self.hgcns[0](seq, adjs, sparse)
        h.append(x)
        if self.hards:
            x, hh2 = self.hgcns[1](seq, adjs, sparse)
            h.append(x)

        for i in range(self.P):
            if False: #self.hards:
                h1 = torch.cat( (hh[i].unsqueeze(0), hh2[i].unsqueeze(0)), dim=-1)
            else:
                h1 = hh[i].unsqueeze(0)
            h_all.append(h1.detach())
        #     print(h1.size())
        # print(h_all)
        return torch.cat(h,dim=-1).detach(), h_all
        
            
        

