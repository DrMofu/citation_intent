#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import pickle
import collections

import json

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


# In[2]:


import transformers
from transformers import AutoTokenizer,AutoModel


# In[3]:


import torch
torch.cuda.is_available()


# In[4]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)


# ## read data

# ### read graph data

# In[5]:


CITATION_FILE = "../data/citation.txt"


# In[6]:


# citation features
with open("../data/one_hop_intent_info.pkl","rb") as f:
    one_hop_intent_info = pickle.load(f) # include zero_hop_intent_info
    
with open("../data/zero_hop_intent_info.pkl","rb") as f:
    zero_hop_intent_info = pickle.load(f)


# In[7]:


entities_emb = np.zeros((len(one_hop_intent_info),8))


# In[8]:


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

r2id = {'background': 1, 'result': 2, 'method': 0}


# ### read sentences

# In[9]:


bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

for param in bert.parameters():
    param.requires_grad = False


# In[10]:


sci_train_data = []
with open("../dataset/scicite/train.jsonl", "r") as f:
    sci_train_data = f.readlines()

sci_test_data = []
with open("../dataset/scicite/test.jsonl", "r") as f:
    sci_test_data = f.readlines()

sci_dev_data = []
with open("../dataset/scicite/dev.jsonl", "r") as f:
    sci_dev_data = f.readlines()

sci_data = []
sci_data.extend([json.loads(item) for item in sci_train_data])
sci_data.extend([json.loads(item) for item in sci_test_data])
sci_data.extend([json.loads(item) for item in sci_dev_data])


# In[11]:


with open("../data/zero_hash_to_corpusID.pkl","rb") as f:
    zero_hash_to_corpusID = pickle.load(f)


# In[12]:


label_2_id = {'background': 1, 'result': 2, 'method': 0} #{"background":0,"method":1,"result":2}

citation_id_to_data = {}
for item in tqdm(sci_data):
    citing_id = zero_hash_to_corpusID[item['citingPaperId']] if item['citingPaperId'] in zero_hash_to_corpusID else -1
    cited_id = zero_hash_to_corpusID[item['citedPaperId']] if item['citedPaperId'] in zero_hash_to_corpusID else -1
    if citing_id==-1 or cited_id==-1:
        continue
    citation_id_to_data[(citing_id,cited_id)] = {"sentence":item["string"],"label":label_2_id[item['label']]}
    with torch.no_grad():
        encoded_input = tokenizer(item["string"], padding=True, truncation=True, return_tensors='pt',max_length=512).to(device)
        output = bert(**encoded_input).last_hidden_state[:, 0, :][0]
        del encoded_input
    citation_id_to_data[(citing_id,cited_id)]["bert_embedding"] = output
        
    tmp_features = np.zeros(16)
    tmp_features[:8]=entities_emb[e2id[citing_id]]
    tmp_features[8:]=entities_emb[e2id[cited_id]]
    citation_id_to_data[(citing_id,cited_id)]["feature"] = torch.tensor(tmp_features).float()
        


# In[13]:


with open("citation_id_to_data.pkl","wb") as f:
    pickle.dump(citation_id_to_data,f)
    
with open("citation_id_to_data.pkl","rb") as f:
    citation_id_to_data = pickle.load(f)


# In[14]:


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
            X.append((citing_paper,cited_paper))
            y.append(label)
            line = f.readline()
    return X,y

X_train, Y_train  = get_labeled_dataset("../dataset/scicite_resplit/train.txt",e2id,r2id)
X_test, Y_test = get_labeled_dataset("../dataset/scicite_resplit/test.txt",e2id,r2id)
X_dev, Y_dev = get_labeled_dataset("../dataset/scicite_resplit/valid.txt",e2id,r2id)


# ## model

# In[16]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import dgl
import itertools
import time

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score


# ### scibert

# In[55]:


class MyDataWrapper(Dataset):  # memory friendly method  
    def __init__(self, X, Y, citation_id_to_data):
        self.X = X
        self.Y = Y
        self.citation_id_to_data = citation_id_to_data
      
    def __len__(self):
        return len(self.Y)
  
    def __getitem__(self, index):
        return citation_id_to_data[self.X[index]]["sentence"], self.Y[index]

train_warp = MyDataWrapper(X_train, Y_train,citation_id_to_data)
test_warp = MyDataWrapper(X_test, Y_test,citation_id_to_data)
dev_warp = MyDataWrapper(X_dev, Y_dev,citation_id_to_data)

train_generator = DataLoader(train_warp,batch_size=8,shuffle=True)
test_generator = DataLoader(test_warp,batch_size=8,shuffle=True)
dev_generator = DataLoader(dev_warp,batch_size=8,shuffle=True)


# In[56]:


def training(model,data_generator,loss_fn,optimizer,epoch=0):
    model.train()
    train_loss = 0
    right_counter = 0
    finished_samples = 0
    with tqdm(data_generator, unit="batch") as tepoch:
        for batch_x, batch_y in tepoch:
            batch_y = batch_y.to(device)
            tepoch.set_description(f"Epoch {epoch}")
            batch_y_pred = model(batch_x)
            loss = loss_fn(batch_y_pred,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*len(batch_x)

            _, batch_pred_index = torch.max(batch_y_pred,1)
            right_counter += sum(batch_y==batch_pred_index).cpu().numpy()
            finished_samples+=len(batch_y)
            tepoch.set_postfix(loss=train_loss/finished_samples,accuracu=100.*right_counter/finished_samples)
        train_loss /= len(data_generator.dataset)
        train_acc = right_counter/len(data_generator.dataset)
        return train_loss,train_acc
    
def evaluation(model,data_generator,loss_fn):  
    model.eval()  
    eval_loss = 0
    y_predict = []
    y_true = []
    with torch.no_grad():
        for batch_x,batch_y in data_generator:
            batch_y = batch_y.to(device)
            batch_y_pred = model(batch_x)
            _, batch_pred_index = torch.max(batch_y_pred,1)
            y_predict.extend(batch_pred_index)
            y_true.extend(batch_y)
            loss = loss_fn(batch_y_pred,batch_y)
            eval_loss += loss.item()*len(batch_y)
    eval_loss /= len(data_generator.dataset)
    return eval_loss,torch.tensor(y_true),torch.tensor(y_predict)


# In[57]:


class MyModel(nn.Module):
    def __init__(self,dropout=0.1,device=device):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.W1 = nn.Linear(768, 128)
        self.W2 = nn.Linear(128, 3)
        self.dropout = torch.nn.Dropout(dropout)
        self.device = device
        
        
    def forward(self, input_sentence):
        encoded_input = self.tokenizer(input_sentence, padding=True, truncation=True, return_tensors='pt',max_length=512).to(self.device)
        x = self.bert(**encoded_input).last_hidden_state[:, 0, :]
        x = self.dropout(x)
        x = self.W1(x)        
        x = torch.nn.functional.relu(x)        
        x = self.W2(x)       
        return x
model = MyModel().to(device)
for param in model.bert.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam(model.parameters())
loss_fn  = torch.nn.CrossEntropyLoss()


# In[58]:


for epoch in range(10):
    train_loss,train_acc = training(model,train_generator,loss_fn,optimizer,epoch)
    train_loss,y_true_train,y_predict_train = evaluation(model,train_generator,loss_fn)
    dev_loss,y_true_dev,y_predict_dev = evaluation(model,dev_generator,loss_fn)
    test_loss,y_true_test,y_predict_test = evaluation(model,test_generator,loss_fn)
    
    print('Epoch: {} [Train] Loss: {:.4f} A: {:.2f} P: {:.2f} R: {:.2f} F1: {:.2f} [Dev] Loss: {:.4f} A: {:.2f} P: {:.2f} R: {:.2f} F1: {:.2f} [Test] Loss: {:.4f} A: {:.2f} P: {:.2f} R: {:.2f} F1: {:.2f}'.format(            
        epoch +1, 
        train_loss,
        accuracy_score(y_true_train,y_predict_train)*100,
        precision_score(y_true_train,y_predict_train,average="macro")*100,
        recall_score(y_true_train,y_predict_train,average="macro")*100,
        f1_score(y_true_train,y_predict_train,average="macro")*100,
        
        dev_loss,
        accuracy_score(y_true_dev,y_predict_dev)*100,
        precision_score(y_true_dev,y_predict_dev,average="macro")*100,
        recall_score(y_true_dev,y_predict_dev,average="macro")*100,
        f1_score(y_true_dev,y_predict_dev,average="macro")*100,
        
        test_loss,
        accuracy_score(y_true_test,y_predict_test)*100,
        precision_score(y_true_test,y_predict_test,average="macro")*100,
        recall_score(y_true_test,y_predict_test,average="macro")*100,
        f1_score(y_true_test,y_predict_test,average="macro")*100
    ))


# ### SCIBERT+Graph

# ### build graph

# In[26]:


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
        
total_link_length = len(graph_head)

g = dgl.graph((graph_head, graph_tail), num_nodes=len(e2id))
g.ndata['embedding'] = torch.tensor(entities_emb,dtype=torch.float32)
# save_graphs("./citation_graph.bin", [g], None)


# glist , _ = load_graphs("./citation_graph.bin")
# g = glist[0]


# In[27]:


g = g.to(device)


# In[51]:


class MyDataWrapper(Dataset):  # memory friendly method  
    def __init__(self, X, Y, citation_id_to_data):
        self.X = X
        self.Y = Y
        self.citation_id_to_data = citation_id_to_data
      
    def __len__(self):
        return len(self.Y)
  
    def __getitem__(self, index):
        return citation_id_to_data[self.X[index]]["bert_embedding"], citation_id_to_data[self.X[index]]["feature"], self.Y[index], [e2id[self.X[index][0]],e2id[self.X[index][1]]]

train_warp = MyDataWrapper(X_train, Y_train,citation_id_to_data)
test_warp = MyDataWrapper(X_test, Y_test,citation_id_to_data)
dev_warp = MyDataWrapper(X_dev, Y_dev,citation_id_to_data)

train_generator = DataLoader(train_warp,batch_size=8,shuffle=True)
test_generator = DataLoader(test_warp,batch_size=8,shuffle=True)
dev_generator = DataLoader(dev_warp,batch_size=8,shuffle=True)


# In[52]:


def training(model,data_generator,loss_fn,optimizer,epoch=0):
    model.train()
    train_loss = 0
    right_counter = 0
    finished_samples = 0
    with tqdm(data_generator, unit="batch") as tepoch:
        for batch_x,batch_f,batch_y,batch_row_ids in tepoch:
            batch_y = batch_y.to(device)
            batch_f = batch_f.to(device)
            tepoch.set_description(f"Epoch {epoch}")
            batch_y_pred = model(batch_x,batch_f,batch_row_ids,g)
            loss = loss_fn(batch_y_pred,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*len(batch_x)

            _, batch_pred_index = torch.max(batch_y_pred,1)
            right_counter += sum(batch_y==batch_pred_index).cpu().numpy()
            finished_samples+=len(batch_y)
            tepoch.set_postfix(loss=train_loss/finished_samples,accuracu=100.*right_counter/finished_samples)
        train_loss /= len(data_generator.dataset)
        train_acc = right_counter/len(data_generator.dataset)
    return train_loss,train_acc
        
def evaluation(model,data_generator,loss_fn):  
    model.eval()  
    eval_loss = 0
    y_predict = []
    y_true = []
    with torch.no_grad():
        for batch_x, batch_f, batch_y ,batch_row_ids in tqdm(data_generator):
            batch_y = batch_y.to(device)
            batch_f = batch_f.to(device)
            
            batch_y_pred = model(batch_x,batch_f,batch_row_ids,g)
            _, batch_pred_index = torch.max(batch_y_pred,1)
            y_predict.extend(batch_pred_index)
            y_true.extend(batch_y)
            loss = loss_fn(batch_y_pred,batch_y)
            eval_loss += loss.item()*len(batch_y)
    eval_loss /= len(data_generator.dataset)
    return eval_loss,torch.tensor(y_true),torch.tensor(y_predict)


# In[38]:


from dgl.nn import SAGEConv

class SciBERT_Graph(nn.Module):
    def __init__(self,dropout=0.0,device=device):
        super(SciBERT_Graph, self).__init__()
        self.W1 = nn.Linear(768, 128)
        self.W2 = nn.Linear(128+16, 16)
        self.W3 = nn.Linear(16, 3)
        self.dropout = torch.nn.Dropout(dropout)
        self.device = device
        self.conv1 = SAGEConv(8, 8, 'mean')
        
    def forward(self, bert_embedding, features, row_ids, g):
        relearned_features = self.conv1(g, g.ndata['embedding'])
        x = self.dropout(bert_embedding)
        x = self.W1(x)        
        x = torch.nn.functional.relu(x)        
        head_node_relearned_features = relearned_features[row_ids[0]]
        tail_node_relearned_features = relearned_features[row_ids[1]]
        x = torch.cat((x,head_node_relearned_features,tail_node_relearned_features),1)
        x = self.W2(x)       
        x = torch.nn.functional.relu(x)        
        x = self.W3(x)      
        return x

model = SciBERT_Graph().to(device)

optimizer = torch.optim.Adam(model.parameters())
loss_fn  = torch.nn.CrossEntropyLoss()


# In[39]:


best_f1 = 0
f1_records = []
best_f1_epoch = -1
EPOCH_NUM = 10
for epoch in range(1,1+EPOCH_NUM):
    train_loss,train_acc = training(model,train_generator,loss_fn,optimizer,epoch)
    dev_loss,y_true_dev,y_predict_dev = evaluation(model,dev_generator,loss_fn)
    test_loss,y_true_test,y_predict_test = evaluation(model,test_generator,loss_fn)
    
    current_f1 = f1_score(y_true_test,y_predict_test,average="macro")*100
    f1_records.append(current_f1)
    if current_f1>best_f1:
        best_f1 = current_f1
        best_f1_epoch = epoch
    print('Epoch: {} [Dev] Loss: {:.4f} A: {:.2f} P: {:.2f} R: {:.2f} F1: {:.2f} [Test] Loss: {:.4f} A: {:.2f} P: {:.2f} R: {:.2f} F1: {:.2f}'.format(            
        epoch, 

        dev_loss,
        accuracy_score(y_true_dev,y_predict_dev)*100,
        precision_score(y_true_dev,y_predict_dev,average="macro")*100,
        recall_score(y_true_dev,y_predict_dev,average="macro")*100,
        f1_score(y_true_dev,y_predict_dev,average="macro")*100,
        
        test_loss,
        accuracy_score(y_true_test,y_predict_test)*100,
        precision_score(y_true_test,y_predict_test,average="macro")*100,
        recall_score(y_true_test,y_predict_test,average="macro")*100,
        f1_score(y_true_test,y_predict_test,average="macro")*100
    ))
    


# In[ ]:





# In[ ]:




