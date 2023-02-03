#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import pickle
import collections
from tqdm import tqdm


# In[2]:


import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F


from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score


# In[3]:


CITATION_FILE = "../data/citation.txt"


# ### read embedding, generate dataset
# 

# In[4]:


# citation features
with open("../data/one_hop_intent_info.pkl","rb") as f:
    one_hop_intent_info = pickle.load(f)
    
with open("../data/zero_hop_intent_info.pkl","rb") as f:
    zero_hop_intent_info = pickle.load(f)


# In[5]:


entities_emb = np.zeros((len(one_hop_intent_info),8))


# In[6]:


entities = list(one_hop_intent_info.keys())

e2id = {}
for i, entity in enumerate(entities):
    e2id[entity] = i
    emb = one_hop_intent_info[entity]
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

r2id = {'background': 1, 'result': 2, 'method': 0, 'methodology': 0}


# In[9]:


def get_labeled_dataset(path,e2id,r2id):
    X = []
    y = []
    with open(path,"r") as f:
        line = f.readline()
        while line:
            split_data = line.split("\t")
            citing_paper = int(split_data[0])
            cited_paper = int(split_data[2])
            label = r2id[split_data[1]]
            citing_id = e2id[citing_paper] if citing_paper in e2id else random.randint(0,entities_emb.shape[0]-1)
            cited_id = e2id[cited_paper] if cited_paper in e2id else random.randint(0,entities_emb.shape[0]-1)
            citing_paper_embedding = entities_emb[citing_id]
            cited_paper_embedding = entities_emb[cited_id]
            X.append(np.concatenate((citing_paper_embedding,cited_paper_embedding),axis=0))
            y.append(label)
            line = f.readline()
    return X,y


# In[10]:


X_train, Y_train  = get_labeled_dataset("../dataset/scicite_origin/train_remove_test.txt",e2id,r2id)
X_test, Y_test = get_labeled_dataset("../dataset/scicite_origin/test.txt",e2id,r2id)


# ### Model

# In[9]:


import torch
from torch.utils.data import DataLoader, Dataset
import time


# In[10]:


INPUT_EMBEDDING_SIZE = 8


# In[25]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)


# In[35]:


class WeaklyLabelDataWrapper(Dataset):  # memory friendly method  
    def __init__(self, data):
        self.data = data
      
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        item = self.data[index]
        y = item[2]
        citing_corpus_id = item[0]
        cited_corpus_id = item[1]
        citing_id = e2id[citing_corpus_id]
        cited_id = e2id[cited_corpus_id]
        citing_emb = entities_emb[citing_id]
        cited_emb = entities_emb[cited_id]
        x = np.concatenate((citing_emb,cited_emb),axis=0)
        return x, y
    
class HumanLabelDataWrapper(Dataset):  # memory friendly method  
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
      
    def __len__(self):
        return len(self.Y)
  
    def __getitem__(self, index):
        return self.X[index], self.Y[index]


# In[118]:


weakly_label_warp = WeaklyLabelDataWrapper(weakly_labeled_data)
weakly_label_generator =  DataLoader(weakly_label_warp,batch_size=1024,shuffle=True)

weakly_label_clean_warp = WeaklyLabelDataWrapper(weakly_labeled_data_clean)
weakly_label_clean_generator =  DataLoader(weakly_label_clean_warp,batch_size=128,shuffle=True)

human_train_warp = HumanLabelDataWrapper(X_train,Y_train)
human_test_warp = HumanLabelDataWrapper(X_test,Y_test)
human_train_generator = DataLoader(human_train_warp,batch_size=8,shuffle=True)
human_test_generator = DataLoader(human_test_warp,batch_size=8,shuffle=True)


# In[60]:


def training(model,data_generator,loss_fn,optimizer):
    model.train()
    train_loss = 0
    right_counter = torch.tensor(0).to(device)
    for batch_x, batch_y in data_generator:
        batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device)
        batch_y_pred = model(batch_x)
        loss = loss_fn(batch_y_pred,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*batch_x.size(0)        
        
        _, batch_pred_index = torch.max(batch_y_pred,1)
        right_counter += sum(batch_y==batch_pred_index)
    train_loss /= len(data_generator.dataset)
    train_acc = right_counter/len(data_generator.dataset)
    return train_loss,train_acc

def evaluation(model,data_generator,loss_fn):  
    model.eval()  
    eval_loss = 0
    y_predict = []
    y_true = []
    with torch.no_grad():
        for batch_x, batch_y in data_generator:
            batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device)
            batch_y_pred = model(batch_x)
            _, batch_pred_index = torch.max(batch_y_pred,1)
            y_predict.extend(batch_pred_index)
            y_true.extend(batch_y)
            loss = loss_fn(batch_y_pred,batch_y)
            eval_loss += loss.item()*batch_x.size(0)
    eval_loss /= len(data_generator.dataset)
    return eval_loss,y_true,y_predict


# In[96]:


class MLP(torch.nn.Module):
    def __init__(self,input_size=INPUT_EMBEDDING_SIZE, dropout=0.0):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_size*2, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 3)
        self.dropout = torch.nn.Dropout(dropout) # for linear layer
      
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        return x
    
    
# class MLP(torch.nn.Module):
#     def __init__(self,input_size=INPUT_EMBEDDING_SIZE, dropout=0.0):
#         super(MLP, self).__init__()
#         self.linear1 = torch.nn.Linear(input_size*2, 128)
#         self.linear2 = torch.nn.Linear(128, 128)
#         self.linear3 = torch.nn.Linear(128, 64)
#         self.linear4 = torch.nn.Linear(64, 3)
#         self.dropout = torch.nn.Dropout(dropout) # for linear layer
      
#     def forward(self, x):
#         x = self.dropout(x)      
#         x = torch.nn.functional.relu(self.linear1(x))
#         x = self.dropout(x)
#         x = torch.nn.functional.relu(self.linear2(x))
#         x = self.dropout(x)
#         x = torch.nn.functional.relu(self.linear3(x))
#         x = self.dropout(x)
#         x = self.linear4(x)
#         return x


# In[99]:


result_dict = {}
for DROPOUT in [0.0,0.1,0.2,0.3,0.4,0.5,0.8]:
    test_max_f1 = 0
    test_max_epoch = 0
    Basic_linear_model = MLP(INPUT_EMBEDDING_SIZE,DROPOUT).to(device)
    optimizer = torch.optim.Adam(Basic_linear_model.parameters())
    loss_fn  = torch.nn.CrossEntropyLoss()
    for i in range(20):
        start_time = time.time()
        train_loss,train_acc = training(Basic_linear_model,human_train_generator,loss_fn,optimizer)
        end_time = time.time()
        train_loss,y_true_train,y_predict_train = evaluation(Basic_linear_model,human_train_generator,loss_fn)
        test_loss,y_true,y_predict = evaluation(Basic_linear_model,human_test_generator,loss_fn)

        y_true_train = list(map(lambda x:x.cpu(),y_true_train))
        y_predict_train = list(map(lambda x:x.cpu(),y_predict_train))

        y_true = list(map(lambda x:x.cpu(),y_true))
        y_predict = list(map(lambda x:x.cpu(),y_predict))
        print('Epoch: {} [Train] Loss: {:.4f} A: {:.2f} P: {:.2f} R: {:.2f} F1: {:.2f} [Test] Loss: {:.4f} A: {:.2f} P: {:.2f} R: {:.2f} F1: {:.2f}'.format(            
            i+1, 
            train_loss,
            accuracy_score(y_true_train,y_predict_train)*100,
            precision_score(y_true_train,y_predict_train,average="macro")*100,
            recall_score(y_true_train,y_predict_train,average="macro")*100,
            f1_score(y_true_train,y_predict_train,average="macro")*100,

            test_loss,
            accuracy_score(y_true,y_predict)*100,
            precision_score(y_true,y_predict,average="macro")*100,
            recall_score(y_true,y_predict,average="macro")*100,
            f1_score(y_true,y_predict,average="macro")*100
        ))
        current_f1 = f1_score(y_true,y_predict,average="macro")*100
        if current_f1>test_max_f1:
            test_max_f1 = current_f1
            test_max_epoch = i+1
    result_dict[DROPOUT] = (test_max_f1,test_max_epoch)
print(result_dict)


# In[100]:


result_dict


# ### GraphSAGE

# In[11]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import itertools
import time

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score


# In[15]:


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
            citing_id = e2id[citing_paper] if citing_paper in e2id else random.randint(0,entities_emb.shape[0]-1)
            cited_id = e2id[cited_paper] if cited_paper in e2id else random.randint(0,entities_emb.shape[0]-1)
            head.append(citing_id)
            tail.append(cited_id)
            y.append(label)
            line = f.readline()
    return head,tail,y


# In[16]:


def test_func(epoch_num = 100,inductive = False):
    max_test_f1 = 0
    max_test_f1_epoch = -1
    for epoch in range(epoch_num):
        model.train()
        pred.train()
        start_time = time.time()
        # forward
        hidden = model(g,g.ndata['embedding'])
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
                
                if inductive:
#                     print("recalculate hidden because the inductive setting")
                    hidden = model(inductive_graph,inductive_graph.ndata["embedding"])
                    predict_result = pred(test_g, hidden)
                    predict_y = predict_result.argmax(1)
                    true_y = test_y_label 
                    print('Inductive TEST ACC:[{:.2f}]  P:[{:.2f}]  R:[{:.2f}]  F1:[{:.2f}]'.format(accuracy_score(true_y,predict_y)*100,
                    precision_score(true_y,predict_y,average="macro")*100,
                    recall_score(true_y,predict_y,average="macro")*100,
                    f1_score(true_y,predict_y,average="macro")*100))


                test_f1 = f1_score(true_y,predict_y,average="macro")*100
                if test_f1>max_test_f1:
                    max_test_f1 = test_f1
                    max_test_f1_epoch = epoch

                print(max_test_f1_epoch,max_test_f1)


# In[17]:


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


# In[18]:


train_head,train_tail,train_y = generate_graph("../dataset/scicite_origin/train_remove_test.txt",e2id,r2id)
test_head,test_tail,test_y = generate_graph("../dataset/scicite_origin/test.txt",e2id,r2id)


# In[19]:


graph_head = []
graph_tail = []
test_head_set = set(test_head) # used to generate inductive graph
counter = 0  # used to generate inductive graph
remove_edge_id_for_inductive = []
with open(CITATION_FILE,"r") as f:
    line = f.readline()
    while line:
        split_data = line.strip().split("\t")
        head_node = int(split_data[0])
        tail_node = int(split_data[2])
        if head_node in e2id and tail_node in e2id:
            graph_head.append(e2id[head_node])
            graph_tail.append(e2id[tail_node])
            if tail_node in test_head_set: # in the inductive setting mode, the new paper should't be cited
                remove_edge_id_for_inductive.append(counter)
            counter+=1
        line = f.readline()
        
len(graph_head)


# In[20]:


g = dgl.graph((graph_head, graph_tail), num_nodes=len(e2id))
inductive_graph = dgl.remove_edges(g, torch.tensor(remove_edge_id_for_inductive))


# In[21]:


# stupid way to set inductive embedding
inductive_emb = entities_emb.copy()
for paper_id in test_head:
    inductive_emb[paper_id][1:] = 0
    inductive_emb[paper_id][4] = -1
inductive_emb = torch.tensor(inductive_emb, dtype=torch.float32)


g = dgl.to_bidirected(g)
inductive_graph = dgl.to_bidirected(inductive_graph)

g.ndata['embedding'] = torch.tensor(entities_emb, dtype=torch.float32)
inductive_graph.ndata['embedding'] = inductive_emb


# In[26]:


train_y_label = torch.tensor(train_y)
test_y_label = torch.tensor(test_y)

train_g = dgl.graph((train_head, train_tail), num_nodes=g.number_of_nodes())
test_g = dgl.graph((test_head, test_tail), num_nodes=g.number_of_nodes())


# In[27]:


# 1 layer GraphSAGE, bidirection, fully inductive setting (no in degree edge, no features)
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        return h

model = GraphSAGE(g.ndata['embedding'].shape[1], 100)
pred = MLPPredictor(100,3)
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
loss = nn.CrossEntropyLoss()
test_func(inductive=True)


# In[ ]:


test_func(inductive=True)


# In[28]:


# two layer
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
model = GraphSAGE(g.ndata['embedding'].shape[1], 100)
pred = MLPPredictor(100,3)
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
loss = nn.CrossEntropyLoss()
test_func(inductive=True)


# In[29]:


test_func(inductive=True)


# In[30]:


# add self loop
g = dgl.add_self_loop(g)


# In[31]:


# 1 layer GraphSAGE, bidirection, fully inductive setting (no in degree edge, no features)
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        return h

model = GraphSAGE(g.ndata['embedding'].shape[1], 100)
pred = MLPPredictor(100,3)
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
loss = nn.CrossEntropyLoss()
test_func(inductive=True)


# In[32]:


test_func(inductive=True)


# In[30]:


train_head,train_tail,train_y = generate_graph("../dataset/scicite_resplit/train.txt",e2id,r2id)
test_head,test_tail,test_y = generate_graph("../dataset/scicite_resplit/test.txt",e2id,r2id)

train_y_label = torch.tensor(train_y)
test_y_label = torch.tensor(test_y)

train_g = dgl.graph((train_head, train_tail), num_nodes=g.number_of_nodes())
test_g = dgl.graph((test_head, test_tail), num_nodes=g.number_of_nodes())


# In[31]:


# 1 layer GraphSAGE, bidirection, fully inductive setting (no in degree edge, no features)
model = GraphSAGE(g.ndata['embedding'].shape[1], 100)
pred = MLPPredictor(100,3)
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
loss = nn.CrossEntropyLoss()
test_func(inductive=True)


# In[ ]:




