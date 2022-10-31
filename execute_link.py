import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import os
import glob
from models import HEMI, GIC, HDGI, HGIC, DGI, DMGI
from utils import process, clustering
import scipy.sparse as sp
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import (v_measure_score, homogeneity_score,
                                     completeness_score, adjusted_rand_score, normalized_mutual_info_score)

import argparse
import random 

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)


def val_loss(edges_pos, edges_neg, embeds, adj_sparse, adj_oris):
    
    if args.model == 'gic' or args.model == 'dgi' or args.model == 'dmgi' or args.model == 'hemi':
        embs = []
        for em in embeds:
            embs.append(em[0, :])#.cpu().detach().numpy())
    else:
        embs = embeds[0, :]#.cpu().detach().numpy()
    #embs = embs / embs.norm(dim=1)[:, None]

    rocs = 0
    aucs = 0
    if args.model == 'dgi' or args.model == 'gic' or args.model == 'dmgi' or args.model == 'hemi':
        for (i,adj_t) in enumerate(adj_oris):
            scoring = embs[i] @ embs[i].t()
            sc_roc, sc_ap = get_roc_score(Test_edges[i], Test_edges_false[i], scoring.cpu().detach().numpy(), adj_t.tocsr())
            rocs+=sc_roc
            aucs+=sc_ap
            #print('i',i, 'AUC', sc_roc, 'AP', sc_ap)
    else:
        for (i,adj_t) in enumerate(adj_oris):
            scoring = embs @ embs.t()
            sc_roc, sc_ap = get_roc_score(Test_edges[i], Test_edges_false[i], scoring.cpu().detach().numpy(), adj_t.tocsr())
            #print(beta, K, alpha, sc_roc, sc_ap,flush=True)
            rocs+=sc_roc
            aucs+=sc_ap
            #print('i',i, 'AUC', sc_roc, 'AP', sc_ap)
                        
    return rocs+aucs
        

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--d', dest='dataset', type=str, default='imdb',help='')
parser.add_argument('--m', dest='model', type=str, default='mi',help='')

EPS = 1e-8
MAX_LOGSTD = 10
class Recon_Loss(torch.nn.Module):
    def decoder(self, z, edge_index):

            value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
            return torch.sigmoid(value)

    def forward(self, z, pos_edge_index, neg_edge_index):

            pos_loss = -torch.log(
                self.decoder(z, pos_edge_index) + EPS).mean()

            """
            pos_edge_index, _ = remove_self_loops(pos_edge_index)
            pos_edge_index, _ = add_self_loops(pos_edge_index)
            if neg_edge_index is None:
                neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
            """
            #neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
            neg_loss = -torch.log(1 -
                                  self.decoder(z, neg_edge_index) +
                                  EPS).mean()

            return pos_loss + neg_loss
    
def get_roc_score(edges_pos, edges_neg, embeddings, adj_sparse):
    "from https://github.com/tkipf/gae"
    
    score_matrix = np.dot(embeddings, embeddings.T)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        pos.append(adj_sparse[edge[0], edge[1]]) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        neg.append(adj_sparse[edge[0], edge[1]]) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    
    #print(preds_all, labels_all )
    
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score


args = parser.parse_args()
dataset = args.dataset
# training params
batch_size = 1
nb_epochs = 2000
patience = 50
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 256 #output of the GCN dimension
shid = 16 #input dimension of semantic-level attention
sparse = False
nonlinearity = 'prelu' # special name to separate parameters

if torch.cuda.is_available():
 device = torch.device("cuda")
 torch.cuda.set_device(0)
else:
 device = torch.device("cpu")
 
    
case = 33




import itertools


seeds = [123]
lams = [-1, 0, 0.25, 0.5, 0.75, 1]
#lams = [-1, 0]
#print('====================',args.model,'================')
for ss in range(len(seeds)):

        
    #adjs, adj0, features, labels, labels_y, idx_train, idx_val, idx_test = process.load_data(path="../data/ACM/", dataset="ACM", frac=0.2)
    #adjs, adj0, features, labels, labels_y, idx_train, idx_val, idx_test = process.load_data(path="../data/DBLP/", dataset="DBLP", frac=0.2)

    if dataset == 'imdb':
        adjs, adj0, features, labels2, labels_y = process.load_data()
        #adjs, adj0, features, labels, labels_y, idx_train, idx_val, idx_test = process.load_data(frac = 0.8)
    elif dataset == 'dblp':
        adjs, adj0, features, labels2, labels_y = process.load_data(path="./data/DBLP/", dataset="DBLP")
        #adjs, adj0, features, labels, labels_y, idx_train, idx_val, idx_test = process.load_data(path="../data/DBLP/", dataset="DBLP", frac=0.8)
    elif dataset == 'acm':
        adjs, adj0, features, labels2, labels_y = process.load_data(path="./data/ACM/", dataset="ACM")
        #adjs, adj0, features, labels, labels_y, idx_train, idx_val, idx_test = process.load_data(path="../data/ACM/", dataset="ACM", frac=0.8)

    #adjs, adj0, features, labels, labels_y, idx_train, idx_val, idx_test = process.load_data(frac = 0.2)

    adj_oris = adjs
    adj_trains = adjs
    for (i,adj_t) in enumerate(adj_oris):
        print(i)
        #print('Edges init',adj.getnnz())

    adjs = []
    edges = {}
    Train_edges = []
    Train_edges_false = []
    Val_edges = []
    Val_edges_false = []
    Test_edges = []
    Test_edges_false = []
    adj_embed = []


    """
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = process.mask_test_edges(adj_t, test_frac=0.2, val_frac=0.0)
    Test_edges.append(test_edges)
    Test_edges_false.append(test_edges_false)    

    adj_oris.append(adj_train)
    adjs = adj_oris
    """
    for adj in adj_trains:
        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = process.mask_test_edges(adj, test_frac=0.2, val_frac=0.05)

        adjs.append(adj_train)
        if torch.cuda.is_available():
            Train_edges.append(torch.LongTensor(train_edges).cuda())
            Train_edges_false.append(torch.LongTensor(train_edges_false).cuda())
        else:
            Train_edges.append(train_edges)
            Train_edges_false.append(train_edges_false)
        Val_edges.append(val_edges)
        Val_edges_false.append(val_edges_false)
        Test_edges.append(test_edges)
        Test_edges_false.append(test_edges_false)


    #"""    




    features, _ = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels2.shape[1]
    P=int(len(adjs))
    print('metapaths:', P)
    nor_adjs = []
    sp_nor_adjs = []
    #adj0 = process.normalize_adj(adj0 + sp.eye(adj0.shape[0]))
    #adj0 = sp.coo_matrix(adj0)
    graph = nx.from_scipy_sparse_matrix(sp.coo_matrix(adj0))
    adj0 = process.normalize_adj(adj0 + sp.eye(adj0.shape[0]))

    if sparse:
        sp_adj0 = process.sparse_mx_to_torch_sparse_tensor(adj0)
        #sp_adj0 = torch.FloatTensor(np.array(sp_adj0))
    else:
        adj0 = (adj0 + sp.eye(adj0.shape[0])).todense() #acm+sp.eye
        adj0 = adj0[np.newaxis]
        adj0 = torch.FloatTensor(np.array(adj0))

    #adjs.append(sp.eye(nb_nodes))

    for adj in adjs:
        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        #adj = process.normalize_adj(adj)

        if sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
            sp_nor_adjs.append(sp_adj)
        else:
            adj = (adj ).todense()
            adj = adj[np.newaxis]
            nor_adjs.append(adj)
        #print(adj.shape)
    features = torch.FloatTensor(features[np.newaxis])
    if sparse:
            sp_nor_adjs = torch.FloatTensor(np.array(sp_nor_adjs))
    else:
            nor_adjs = torch.FloatTensor(np.array(nor_adjs))
    labels = torch.FloatTensor(labels2[np.newaxis])
    for lam in lams:
        print('====================',args.model, dataset,'================')
    
        

        seed = seeds[ss]
        if args.model == 'hemi':
            print('Lam', lam, 'Seed', seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        if args.model == 'hemi':
            model = HEMI(nb_nodes, ft_size, hid_units, shid, P, nonlinearity, lam, dataset)
        elif args.model == 'hdgi':
            model = HDGI(ft_size, hid_units, shid, P, nonlinearity)
        elif args.model == 'dgi':
            model = DGI(ft_size, hid_units, P, nonlinearity)
        elif args.model == 'gic':
            num_clusters = 32
            beta = 10
            alpha = 0.5
            model = GIC(nb_nodes, ft_size, hid_units, P, nonlinearity, num_clusters, beta, alpha)
        elif args.model == 'hgic':
            num_clusters = 32
            beta = 10
            alpha = 0.5
            model = HGIC(nb_nodes, ft_size, hid_units, shid, P, nonlinearity, num_clusters, beta, alpha)
            
        elif args.model == 'dmgi':
            reg = 0.0001
            model = DMGI(nb_nodes, ft_size, hid_units, shid, P, nonlinearity, reg)
            
        if args.model == 'dmgi':
            optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        else:
            optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
        
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)



        if torch.cuda.is_available():
            print('Using CUDA')
            model.cuda()
            features = features.cuda()
            recon_loss = Recon_Loss().cuda()
            if sparse:
                sp_nor_adjs = sp_nor_adjs.cuda()
                sp_adj0 = sp_adj0.cuda()
            else:
                nor_adjs = nor_adjs.cuda()
                adj0 = adj0.cuda()
            labels = labels.cuda()
            #idx_train = idx_train.cuda()
            #idx_val = idx_val.cuda()
            #idx_test = idx_test.cuda()

        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0

        for epoch in range(nb_epochs):
        #for epoch in range(0):
            model.train()
            optimiser.zero_grad()

            idx = np.random.permutation(nb_nodes)
            shuf_fts = features[:, idx, :]

            lbl_1 = torch.ones(batch_size, nb_nodes)
            lbl_2 = torch.zeros(batch_size, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)

            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda()
                lbl = lbl.cuda()

            loss, zs = model(features, shuf_fts, lbl, sp_nor_adjs if sparse else nor_adjs, sparse, None, None, None) 

            if lam == -1:
                loss =0
            for i in range(P):
                loss += (1/P)  * recon_loss(zs[i], Train_edges[i], Train_edges_false[i])

            if loss < best and epoch > 0:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'best_'+args.model+'_'+dataset+'.pkl')
            elif epoch > 0:
                cnt_wait += 1

            if cnt_wait == patience:
                print('Early stopping!')
                break
                
            
            if epoch % 100 ==0 :
                print('Loss:', loss)

            loss.backward()
            optimiser.step()

        print('Loading {}th epoch'.format(best_t))
        model.load_state_dict(torch.load('best_'+args.model+'_'+dataset+'.pkl'))

        with torch.no_grad():
            _, embeds = model.embed(features, sp_nor_adjs if sparse else nor_adjs, sparse, None)


        #if args.model == 'dgi' or args.model == 'gic' or args.model == 'dmgi' or args.model == 'hemi' or args.model == 'hdgi':
        embs = []
        for em in embeds:
            embs.append(em[0, :].cpu().detach().numpy())
        # else:
        #     embs = embeds[0, :].cpu().detach().numpy()
        #embs = embs / embs.norm(dim=1)[:, None]
        
        rocs = 0
        aucs = 0
        #if args.model == 'dgi' or args.model == 'gic' or args.model == 'dmgi' or args.model == 'hemi' or args.model == 'hdgi':
        for (i,adj_t) in enumerate(adj_oris):
            #scoring = embs[i] @ embs[i].t()
            sc_roc, sc_ap = get_roc_score(Test_edges[i], Test_edges_false[i], embs[i], adj_t.tocsr())
            rocs+=sc_roc
            aucs+=sc_ap
            print('i',i, 'AUC', sc_roc, 'AP', sc_ap)
        # else:
        #     for (i,adj_t) in enumerate(adj_oris):
        #         #scoring = embs @ embs.t()
        #         sc_roc, sc_ap = get_roc_score(Test_edges[i], Test_edges_false[i], embs, adj_t.tocsr())
        #         #print(beta, K, alpha, sc_roc, sc_ap,flush=True)
        #         rocs+=sc_roc
        #         aucs+=sc_ap
        #         print('i',i, 'AUC', sc_roc, 'AP', sc_ap)
        print('Tot AUC', rocs/P, 'AP', aucs/P)

    if args.model != 'hemi':
        break