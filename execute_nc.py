import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import os
import glob
from models import HEMI, DGI, LogReg, GIC, HGIC, HDGI, DMGI, MNI_DGI, SSMGRL
from utils import process, clustering
import scipy.sparse as sp
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import (v_measure_score, homogeneity_score,
                                     completeness_score, adjusted_rand_score, normalized_mutual_info_score)

import argparse
import random 
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--d', dest='dataset', type=str, default='imdb',help='')
parser.add_argument('--m', dest='model', type=str, default='hemi',help='')

EPS = 1e-8

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
features, _ = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels2.shape[1]
P=int(len(adjs))

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



import itertools

"""
adj_c = []
for L in range(1, P):
    for sub_adj in itertools.combinations(adjs, L):
        print(sub_adj)
        adj_c.extend(list(sub_adj))
print(len(adj_c))
"""       
#adjs = adj_c
seeds = [123] #, 132, 321, 312, 231]
lams = [0, .25, .5, .75, 1]
print('====================',args.model,'================')
for lam in lams:
    #lams = [0, 0.25, 0.5, 0.75, 1]
    for ss in range(1):
        
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
        elif args.model == 'mni_dgi':
            reg = 0.0001
            model = MNI_DGI(nb_nodes, ft_size, hid_units, shid, P, nonlinearity, reg)
        elif args.model == 'ssmgrl':
            model = SSMGRL(ft_size, hid_units, shid, P, nonlinearity)
            
        if args.model == 'dmgi' or args.model == 'mni_dgi':
            optimiser = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
        else:
            optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)


        if torch.cuda.is_available():
            print('Using CUDA')
            model.cuda()
            features = features.cuda()
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

            loss, _ = model(features, shuf_fts, lbl, sp_nor_adjs if sparse else nor_adjs, sparse, None, None, None) 

            # if args.model == 'dmgi':
            #     loss = b_xent(logits[0], lbl) + logits[1]
            # else:
            #     loss = b_xent(logits, lbl) 

            if epoch % 100 ==0 :
                print('Loss:', loss)

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

            loss.backward()
            optimiser.step()

        print('Loading {}th epoch'.format(best_t))
        model.load_state_dict(torch.load('best_'+args.model+'_'+dataset+'.pkl'))

        model.eval()
        with torch.no_grad():
            embeds, _ = model.embed(features, sp_nor_adjs if sparse else nor_adjs, sparse, None)

        # if args.model == 'dgi' or args.model == 'gic' :
        #     embeds = torch.cat(embeds, -1)

        #for embeds in embeds_all:
        accs = []
        mac_f1 = []
        accs2 = []
        mac2_f1 = []
        nmis = []
        aris = []
        for frac in [0.2]:
            tot = torch.zeros(1)
            tot = tot.cuda()
            tot_mac = 0
            for _ in range(10):
                nb = len(labels2)
                original = range(nb)
                idx_t = random.sample(original, nb)

                # if dataset == 'imdb':
                #     tr_s = 300
                #     val_s = 300+300
                # if dataset == 'acm':
                #     tr_s = 600
                #     val_s = 600+600
                # if dataset == 'dblp':
                #     tr_s = 80
                #    val_s = 80+80
                tr_s = int(frac*nb)
                val_s = int(0.1*nb)+tr_s
                te_s = int(0.1*nb)+val_s

                idx_train = idx_t[:tr_s]
                idx_val = idx_t[tr_s:val_s]
                idx_test = idx_t[val_s:]
                # idx_train = idx_t[:tr_s]
                # idx_val = idx_t[tr_s:val_s]
                # idx_test = idx_t[val_s:te_s]
                print(len(idx_train), len(idx_val), len(idx_test))



                idx_train = torch.LongTensor(idx_train)
                idx_val = torch.LongTensor(idx_val)
                idx_test = torch.LongTensor(idx_test)

                idx_train = idx_train.cuda()
                idx_val = idx_val.cuda()
                idx_test = idx_test.cuda()

                train_embs = embeds[0, idx_train]
                val_embs = embeds[0, idx_val]
                test_embs = embeds[0, idx_test]

                train_lbls = torch.argmax(labels[0, idx_train], dim=1)
                val_lbls = torch.argmax(labels[0, idx_val], dim=1)
                test_lbls = torch.argmax(labels[0, idx_test], dim=1)

                bad_counter = 0
                best = 10000
                loss_values = []
                best_epoch = 0
                patience = 20

                log = LogReg(embeds.size(-1), nb_classes)
                opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
                if torch.cuda.is_available():
                   log.cuda()

                
                log_ep = 10000
                
                for epoch in range(log_ep):
                    log.train()
                    opt.zero_grad()
                    logits = log(train_embs)
                    loss = xent(logits, train_lbls)
                    logits_val = log(val_embs)
                    loss_val = xent(logits_val, val_lbls)
                    loss_values.append(loss_val)
                    loss.backward()
                    opt.step()
                    if epoch > 1:
                        torch.save(log.state_dict(), '{}.mlp.pkl'.format(epoch))
                        if loss_values[-1] < best:
                           best = loss_values[-1]
                           best_epoch = epoch
                           bad_counter = 0
                        else:
                           bad_counter += 1

                        if bad_counter == patience:
                            break

                        files = glob.glob('*.mlp.pkl')
                        for file in files:
                            epoch_nb = int(file.split('.')[0])
                            if epoch_nb < best_epoch:
                               os.remove(file)
                        """
                        if epoch % 50 ==0:
                            logits = log(test_embs)
                            preds = torch.argmax(logits, dim=1)
                            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                            mac = torch.Tensor(np.array(process.macro_f1(preds, test_lbls))) 
                            print(acc, mac)
                        """
                files = glob.glob('*.mlp.pkl')
                for file in files:
                    epoch_nb = int(file.split('.')[0])
                    if epoch_nb > best_epoch:
                        os.remove(file)

                #print("Optimization Finished!")  
                # Restore best model
                #print('Loading {}th epoch'.format(best_epoch))
                
                log.load_state_dict(torch.load('{}.mlp.pkl'.format(best_epoch)))

                files = glob.glob('*.mlp.pkl')
                for file in files:
                        os.remove(file)


                logits = log(test_embs)
                preds = torch.argmax(logits, dim=1)
                #acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                if frac == 0.2:
                    acc = torch.Tensor(np.array(process.micro_f1(preds, test_lbls))) 
                    accs.append(acc)
                    mac = torch.Tensor(np.array(process.macro_f1(preds, test_lbls))) 
                    mac_f1.append(mac)
                    print('Classif 0.2:', acc, mac)
                elif frac == 0.8:
                    acc2 = torch.Tensor(np.array(process.micro_f1(preds, test_lbls))) 
                    accs2.append(acc2)
                    mac2 = torch.Tensor(np.array(process.macro_f1(preds, test_lbls))) 
                    mac2_f1.append(mac2)
                    print('Classif 0.8:', acc2, mac2)






            #print(accs)
            #print(mac_f1)
            embs = embeds[0,:]
            #embs = embs / embs.norm(dim=1)[:, None]
            kmeans_input = embs.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=labels.size(-1), random_state=0).fit(kmeans_input)
            pred = kmeans.predict(kmeans_input)

            nmi = normalized_mutual_info_score(labels_y, pred)
            adjscore = adjusted_rand_score(labels_y, pred)
            nmis.append(nmi)
            aris.append(adjscore)
            print('Clust:', nmi, adjscore)
        #print('NMI', nmi, 'ADJ', adjscore)
        
        #"""
        accs = torch.stack(accs)
        print('Average mic_f1 0.2:',accs.mean(), accs.std())

        mac_f1 = torch.stack(mac_f1)
        print('Average mac_f1 0.2:', mac_f1.mean(), mac_f1.std())
        #"""
        try:
            accs2 = torch.stack(accs2)
            print('Average mic_f1 0.8:',accs2.mean(), accs2.std())

            mac2_f1 = torch.stack(mac2_f1)
            print('Average mac_f1 0.8:', mac2_f1.mean(), mac2_f1.std())
        except:
            pass
        #"""
        
        import statistics
        import statistics
        print('Average nmi:',np.mean(nmis), np.std(nmis))

        
        print('Average adj:', np.mean(aris), np.std(aris))

    print('======')
    if args.model != 'hemi':
        break

    #clustering.my_Kmeans(kmeans_input, labels_y, k=labels.size(-1), time=10, return_NMI=False)
"""
accs = torch.stack(accs)
print('Average mic_f1 0.2:',accs.mean(), accs.std())

mac_f1 = torch.stack(mac_f1)
print('Average mac_f1 0.2:', mac_f1.mean(), mac_f1.std())

accs2 = torch.stack(accs2)
print('Average mic_f1 0.8:',accs2.mean(), accs2.std())

mac2_f1 = torch.stack(mac2_f1)
print('Average mac_f1 0.8:', mac2_f1.mean(), mac2_f1.std())

nmis = torch.stack(nmis)
import statistics
print('Average nmi:',np.mean(nmis), np.std(nmis))

aris = torch.stack(aris)
print('Average adj:', np.mean(aris), np.std(aris))


print('======')
"""
