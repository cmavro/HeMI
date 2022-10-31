import torch
import torch.nn as nn
import torch.nn.functional as F
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        #c_x = c
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
#        print(logits)
        return logits

class Discriminator2(nn.Module):
    def __init__(self, n_h):
        super(Discriminator2, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c_x, h_pl, h_mi, s_bias1=None, s_bias2=None):
        #c_x = torch.unsqueeze(c, 1)
        #c_x = c_x.expand_as(h_pl)
        #c_x = c
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
#        print(logits)
        return logits

class Discriminator_cluster(nn.Module):
    def __init__(self, n_in, n_h , n_nb , num_clusters ):
        super(Discriminator_cluster, self).__init__()
        
        self.n_nb = n_nb
        self.n_h = n_h
        self.num_clusters=num_clusters

    def forward(self, c, c2, h_0, h_pl, h_mi, S, s_bias1=None, s_bias2=None):
        
        c_x = c.expand_as(h_0)
        
        sc_1 =torch.bmm(h_pl.view(self.n_nb, 1, self.n_h), c_x.view(self.n_nb, self.n_h, 1))
        sc_2 = torch.bmm(h_mi.view(self.n_nb, 1, self.n_h), c_x.view(self.n_nb, self.n_h, 1))
        
        
        
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1,sc_2),0).view(1,-1)

                            
        return logits