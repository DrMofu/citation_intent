import pandas as pd
import numpy as np
import random
import pickle
import collections

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import itertools
import time

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

CITATION_FILE = "/nas/ckgfs/pubgraphs/xinweidu/semantic/citations/onehop/known/citation.txt"
DATASET = "Origin" # Resplit

# citation features
with open("../data/one_hop_intent_info.pkl","rb") as f:
    one_hop_intent_info = pickle.load(f) # include zero_hop_intent_info
    

entities = list(one_hop_intent_info.keys())

e2id = {}
for i, entity in enumerate(entities):
    e2id[entity] = i

r2id = {'background': 1, 'result': 2, 'method': 0}

graph_head = []
graph_tail = []
with open(CITATION_FILE,"r") as f:
    line = f.readline()
    while line:
        split_data = line.strip().split("\t")
        head_node = int(split_data[0])
        tail_node = int(split_data[2])
        if head_node in e2id and tail_node in e2id:
            graph_head.append(e2id[head_node])
            graph_tail.append(e2id[tail_node])
        line = f.readline()
g = dgl.graph((graph_head, graph_tail), num_nodes=len(e2id))

def generate_graph(path,e2id,r2id):
    head = []
    tail = []
    y = []
    with open(path,"r") as f:
        line = f.readline()
        while line:
            split_data = line.split("\t")
            citing_paper = int(split_data[0])
            cited_paper = int(split_data[2])
            label = r2id[split_data[1]]
            # citing_id = e2id[citing_paper]
            # cited_id = e2id[cited_paper]
            citing_id = e2id[citing_paper] if citing_paper in e2id else random.randint(0,len(e2id)-1)
            cited_id = e2id[cited_paper] if cited_paper in e2id else random.randint(0,len(e2id)-1)
            head.append(citing_id)
            tail.append(cited_id)
            y.append(label)
            line = f.readline()
    return head,tail,y

if DATASET=="Origin":
    train_head,train_tail,train_y = generate_graph("../dataset/scicite_origin/train.txt",e2id,r2id)
    test_head,test_tail,test_y = generate_graph("../dataset/scicite_origin/test.txt",e2id,r2id)
else:
    train_head,train_tail,train_y = generate_graph("../dataset/scicite_resplit/train.txt",e2id,r2id)
    test_head,test_tail,test_y = generate_graph("../dataset/scicite_resplit/test.txt",e2id,r2id)


train_y_label = torch.tensor(train_y)
test_y_label = torch.tensor(test_y)

train_g = dgl.graph((train_head, train_tail), num_nodes=g.number_of_nodes())
test_g = dgl.graph((test_head, test_tail), num_nodes=g.number_of_nodes())


def test_func(epoch_num = 100, citation_graph = g):
    max_test_f1 = 0
    max_test_f1_epoch = -1
    for epoch in range(epoch_num):
        model.train()
        pred.train()
        start_time = time.time()
        # forward
        hidden = model(citation_graph,citation_graph.ndata['embedding'])
        predict_result = pred(train_g, hidden)
        train_loss = loss(predict_result, train_y_label)

        # backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        end_time = time.time()
        if epoch%1==0:
            print('[Epoch: {}] {:.2f}s\tTraining Loss:{:.4f}'.format(            
                epoch+1, 
                end_time - start_time,
                train_loss,
              ))
            with torch.no_grad():
                model.eval()
                pred.eval()
                predict_result = pred(train_g, hidden)
                predict_y = predict_result.argmax(1)
                true_y = train_y_label
                print('Train ACC:[{:.2f}]  P:[{:.2f}]  R:[{:.2f}]  F1:[{:.2f}]'.format(accuracy_score(true_y,predict_y)*100,
                precision_score(true_y,predict_y,average="macro")*100,
                recall_score(true_y,predict_y,average="macro")*100,
                f1_score(true_y,predict_y,average="macro")*100))

                predict_result = pred(test_g, hidden)
                predict_y = predict_result.argmax(1)
                true_y = test_y_label 
                print('TEST ACC:[{:.2f}]  P:[{:.2f}]  R:[{:.2f}]  F1:[{:.2f}]'.format(accuracy_score(true_y,predict_y)*100,
                precision_score(true_y,predict_y,average="macro")*100,
                recall_score(true_y,predict_y,average="macro")*100,
                f1_score(true_y,predict_y,average="macro")*100))


                test_f1 = f1_score(true_y,predict_y,average="macro")*100
                if test_f1>max_test_f1:
                    max_test_f1 = test_f1
                    max_test_f1_epoch = epoch

                print(max_test_f1_epoch,max_test_f1)
                # confusion = confusion_matrix(true_y,predict_y)/len(true_y)
                # print("confusion matrix:\n",confusion)
    return max_test_f1_epoch,max_test_f1

from dgl.nn import SAGEConv
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
#         self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
#         h = F.relu(h)
#         h = self.conv2(g, h)
        return h

import dgl.function as fn


# # Use MLP (my previous method)
class MLPPredictor(nn.Module):
    def __init__(self, h_feats,output_class,dropout=0.0):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, 64)
        self.W2 = nn.Linear(64, 32)
        self.W3 = nn.Linear(32, output_class)
        self.dropout = torch.nn.Dropout(dropout) # for linear layer

    def apply_edges(self, edges):
        x = torch.cat([edges.src['h'], edges.dst['h']], 1)
#         x = self.dropout(x)
        x = self.W1(x)
        x = F.relu(x)
#         x = self.dropout(x)
        x = self.W2(x)
        x = F.relu(x)
#         x = self.dropout(x)
        x = self.W3(x)
        return {'score': x}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

total_max_test_f1 = []
total_max_test_f1_epoch = []

for k in [0,10,20,30,40,50,60,70,80,90,100]:
    print("---------------------------[{}]-----------------------------".format(k))
    with open("../data/intent_features/corruption_intent/corrupt_{}.pkl".format(k),"rb") as f:
        sub_intent_info = pickle.load(f) # include zero_hop_intent_info
        
    entities_emb = np.zeros((len(one_hop_intent_info),8))
    for key in sub_intent_info:
        entities_emb[e2id[key]] = sub_intent_info[key]

    for i in range(len(one_hop_intent_info)):
        emb = entities_emb[i]
        citing_intent_num = emb[1]+emb[2]+emb[3]
        if citing_intent_num>0:
            emb[1]/=citing_intent_num
            emb[2]/=citing_intent_num
            emb[3]/=citing_intent_num

        cited_intent_num = emb[5]+emb[6]+emb[7]
        if cited_intent_num>0:
            emb[5]/=cited_intent_num
            emb[6]/=cited_intent_num
            emb[7]/=cited_intent_num

        emb[0] = np.log10(emb[0]+1e-1)
        emb[4] = np.log10(emb[4]+1e-1)
        entities_emb[i] = emb
        
    g.ndata['embedding'] = torch.tensor(entities_emb,dtype=torch.float32)
    
    # one layer
    model = GraphSAGE(g.ndata['embedding'].shape[1], 100)
    pred = MLPPredictor(100,3)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
    loss = nn.CrossEntropyLoss()
    max_test_f1_epoch,max_test_f1 = test_func(epoch_num=100)
    
    
    total_max_test_f1.append(max_test_f1)
    total_max_test_f1_epoch.append(max_test_f1_epoch)

print(total_max_test_f1)
print(total_max_test_f1_epoch)