#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle

# import warnings
# warnings.filterwarnings("ignore")


# In[2]:


import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import random
import pickle
import collections

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score


# In[3]:


CITATION_FILE = "../data/citation.txt"

ENTITIES_FILE = '../data/embedding_file/entities.tsv'
RELATIONS_FILE = '../data/embedding_file/relations.tsv'


# In[6]:


entities = list(pd.read_csv(ENTITIES_FILE, header=None, sep='\t', index_col=0)[1])
relations = list(pd.read_csv(RELATIONS_FILE, header=None, sep='\t', index_col=0)[1])

e2id = {}
for i, entity in enumerate(entities):
    e2id[entity] = i

r2id = {'methodology': 0, 'background': 1, 'result': 2, 'method': 0}
id2r = {0: 'method', 1: 'background', 2: 'result'}


# ### read embedding, generate dataset

# In[7]:


ENTITIES_EMB_FILE = '../data/embedding/ComplEx_ComplEx_EPOCH100_EMB100_LR0.3_NEG512_RC1e-06_GAMMA12_ADV0.25_0/ComplEx_EPOCH100_EMB100_LR0.3_NEG512_RC1e-06_GAMMA12_ADV0.25_ComplEx_entity.npy'
RELATIONS_EMB_FILE = '../data/embedding/ComplEx_ComplEx_EPOCH100_EMB100_LR0.3_NEG512_RC1e-06_GAMMA12_ADV0.25_0/ComplEx_EPOCH100_EMB100_LR0.3_NEG512_RC1e-06_GAMMA12_ADV0.25_ComplEx_relation.npy'

entities_emb = np.load(ENTITIES_EMB_FILE)
relations_emb = np.load(RELATIONS_EMB_FILE)[1:]


# In[8]:


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


X_train, Y_train  = get_labeled_dataset("../dataset/scicite_resplit/train.txt",e2id,r2id)
X_dev, Y_dev = get_labeled_dataset("../dataset/scicite_resplit/valid.txt",e2id,r2id)
X_test, Y_test = get_labeled_dataset("../dataset/scicite_resplit/test.txt",e2id,r2id)


# ### ML

# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score


# In[12]:


model_performance = {}

model_SVM = LinearSVC()
model_logistic = LogisticRegression(random_state=0,max_iter=300)
model_NB = BernoulliNB()
model_Tree = DecisionTreeClassifier()


# In[13]:


get_ipython().run_cell_magic('time', '', 'model_SVM.fit(X_train,Y_train)')


# In[15]:


Y_train_predict_SVM = model_SVM.predict(X_train)
Y_test_predict_SVM = model_SVM.predict(X_test)
Y_dev_predict_SVM = model_SVM.predict(X_dev)

model_performance["SVM"] = [accuracy_score(Y_train,Y_train_predict_SVM),precision_score(Y_train,Y_train_predict_SVM,average="macro"),
                   recall_score(Y_train,Y_train_predict_SVM,average="macro"),f1_score(Y_train,Y_train_predict_SVM,average="macro"),
                    accuracy_score(Y_dev,Y_dev_predict_SVM),precision_score(Y_dev,Y_dev_predict_SVM,average="macro"),
                    recall_score(Y_dev,Y_dev_predict_SVM,average="macro"),f1_score(Y_dev,Y_dev_predict_SVM,average="macro"),
                    accuracy_score(Y_test,Y_test_predict_SVM),precision_score(Y_test,Y_test_predict_SVM,average="macro"),
                    recall_score(Y_test,Y_test_predict_SVM,average="macro"),f1_score(Y_test,Y_test_predict_SVM,average="macro")]

print("[MODEL:SVM]")
print("[Training Dataset]")
print("Accuracy: %f" % model_performance["SVM"][0])
print("Precision: %f" % model_performance["SVM"][1])
print("Recall: %f" % model_performance["SVM"][2])
print("F1: %f" % model_performance["SVM"][3])

print("[Dev Dataset]")
print("Accuracy: %f" % model_performance["SVM"][4])
print("Precision: %f" % model_performance["SVM"][5])
print("Recall: %f" % model_performance["SVM"][6])
print("F1: %f" % model_performance["SVM"][7])

print("[Testing Dataset]")
print("Accuracy: %f" % model_performance["SVM"][8])
print("Precision: %f" % model_performance["SVM"][9])
print("Recall: %f" % model_performance["SVM"][10])
print("F1: %f" % model_performance["SVM"][11])


# In[16]:


get_ipython().run_cell_magic('time', '', 'model_logistic.fit(X_train,Y_train)')


# In[17]:


Y_train_predict_logistic = model_logistic.predict(X_train)
Y_test_predict_logistic = model_logistic.predict(X_test)
Y_dev_predict_logistic = model_logistic.predict(X_dev)

model_performance["Logistic"] = [accuracy_score(Y_train,Y_train_predict_logistic),precision_score(Y_train,Y_train_predict_logistic,average="macro"),
                   recall_score(Y_train,Y_train_predict_logistic,average="macro"),f1_score(Y_train,Y_train_predict_logistic,average="macro"),
                    accuracy_score(Y_dev,Y_dev_predict_logistic),precision_score(Y_dev,Y_dev_predict_logistic,average="macro"),
                    recall_score(Y_dev,Y_dev_predict_logistic,average="macro"),f1_score(Y_dev,Y_dev_predict_logistic,average="macro"),
                    accuracy_score(Y_test,Y_test_predict_logistic),precision_score(Y_test,Y_test_predict_logistic,average="macro"),
                    recall_score(Y_test,Y_test_predict_logistic,average="macro"),f1_score(Y_test,Y_test_predict_logistic,average="macro")]

print("[MODEL:Logistic Regression]")
print("[Training Dataset]")
print("Accuracy: %f" % model_performance["Logistic"][0])
print("Precision: %f" % model_performance["Logistic"][1])
print("Recall: %f" % model_performance["Logistic"][2])
print("F1: %f" % model_performance["Logistic"][3])

print("[Dev Dataset]")
print("Accuracy: %f" % model_performance["Logistic"][4])
print("Precision: %f" % model_performance["Logistic"][5])
print("Recall: %f" % model_performance["Logistic"][6])
print("F1: %f" % model_performance["Logistic"][7])

print("[Testing Dataset]")
print("Accuracy: %f" % model_performance["Logistic"][8])
print("Precision: %f" % model_performance["Logistic"][9])
print("Recall: %f" % model_performance["Logistic"][10])
print("F1: %f" % model_performance["Logistic"][11])


# In[18]:


get_ipython().run_cell_magic('time', '', 'model_NB.fit(X_train,Y_train)')


# In[19]:


Y_train_predict_NB = model_NB.predict(X_train)
Y_test_predict_NB = model_NB.predict(X_test)
Y_dev_predict_NB = model_NB.predict(X_dev)

model_performance["Naive_Bayes"] = [accuracy_score(Y_train,Y_train_predict_NB),precision_score(Y_train,Y_train_predict_NB,average="macro"),
                   recall_score(Y_train,Y_train_predict_NB,average="macro"),f1_score(Y_train,Y_train_predict_NB,average="macro"),
                    accuracy_score(Y_dev,Y_dev_predict_NB),precision_score(Y_dev,Y_dev_predict_NB,average="macro"),
                    recall_score(Y_dev,Y_dev_predict_NB,average="macro"),f1_score(Y_dev,Y_dev_predict_NB,average="macro"),
                    accuracy_score(Y_test,Y_test_predict_NB),precision_score(Y_test,Y_test_predict_NB,average="macro"),
                    recall_score(Y_test,Y_test_predict_NB,average="macro"),f1_score(Y_test,Y_test_predict_NB,average="macro")]

print("[MODEL:Naive Bayes]")
print("[Training Dataset]")
print("Accuracy: %f" % model_performance["Naive_Bayes"][0])
print("Precision: %f" % model_performance["Naive_Bayes"][1])
print("Recall: %f" % model_performance["Naive_Bayes"][2])
print("F1: %f" % model_performance["Naive_Bayes"][3])

print("[Dev Dataset]")
print("Accuracy: %f" % model_performance["Naive_Bayes"][4])
print("Precision: %f" % model_performance["Naive_Bayes"][5])
print("Recall: %f" % model_performance["Naive_Bayes"][6])
print("F1: %f" % model_performance["Naive_Bayes"][7])


print("[Testing Dataset]")
print("Accuracy: %f" % model_performance["Naive_Bayes"][8])
print("Precision: %f" % model_performance["Naive_Bayes"][9])
print("Recall: %f" % model_performance["Naive_Bayes"][10])
print("F1: %f" % model_performance["Naive_Bayes"][11])


# In[20]:


get_ipython().run_cell_magic('time', '', 'model_Tree.fit(X_train,Y_train)')


# In[21]:


Y_train_predict_Tree = model_Tree.predict(X_train)
Y_test_predict_Tree = model_Tree.predict(X_test)
Y_dev_predict_Tree = model_Tree.predict(X_dev)

model_performance["Decision_tree"] = [accuracy_score(Y_train,Y_train_predict_Tree),precision_score(Y_train,Y_train_predict_Tree,average="macro"),
                   recall_score(Y_train,Y_train_predict_Tree,average="macro"),f1_score(Y_train,Y_train_predict_Tree,average="macro"),
                    accuracy_score(Y_dev,Y_dev_predict_Tree),precision_score(Y_dev,Y_dev_predict_Tree,average="macro"),
                    recall_score(Y_dev,Y_dev_predict_Tree,average="macro"),f1_score(Y_dev,Y_dev_predict_Tree,average="macro"),
                    accuracy_score(Y_test,Y_test_predict_Tree),precision_score(Y_test,Y_test_predict_Tree,average="macro"),
                    recall_score(Y_test,Y_test_predict_Tree,average="macro"),f1_score(Y_test,Y_test_predict_Tree,average="macro")]

print("[MODEL:Decision Tree]")
print("[Training Dataset]")
print("Accuracy: %f" % model_performance["Decision_tree"][0])
print("Precision: %f" % model_performance["Decision_tree"][1])
print("Recall: %f" % model_performance["Decision_tree"][2])
print("F1: %f" % model_performance["Decision_tree"][3])

print("[Dev Dataset]")
print("Accuracy: %f" % model_performance["Decision_tree"][4])
print("Precision: %f" % model_performance["Decision_tree"][5])
print("Recall: %f" % model_performance["Decision_tree"][6])
print("F1: %f" % model_performance["Decision_tree"][7])

print("[Testing Dataset]")
print("Accuracy: %f" % model_performance["Decision_tree"][8])
print("Precision: %f" % model_performance["Decision_tree"][9])
print("Recall: %f" % model_performance["Decision_tree"][10])
print("F1: %f" % model_performance["Decision_tree"][11])


# ### deep learning

# In[22]:


import torch
from torch.utils.data import DataLoader, Dataset
import time


# In[23]:


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)


# In[24]:


class MyDataWrapper(Dataset):  # memory friendly method  
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
      
    def __len__(self):
        return len(self.Y)
  
    def __getitem__(self, index):
        return self.X[index], self.Y[index]


# In[25]:


train_warp = MyDataWrapper(X_train,Y_train)
dev_warp = MyDataWrapper(X_dev,Y_dev)
test_warp = MyDataWrapper(X_test,Y_test)

train_generator = DataLoader(train_warp,batch_size=8,shuffle=True)
dev_generator = DataLoader(dev_warp,batch_size=8,shuffle=True)
test_generator = DataLoader(test_warp,batch_size=8,shuffle=True)


# In[26]:


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


# In[27]:


class MLP(torch.nn.Module):
    def __init__(self,dropout):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(200*2, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 3)
        self.dropout = torch.nn.Dropout(dropout) # for linear layer
      
    def forward(self, x):
        x = self.dropout(x)      
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
# class MLP(torch.nn.Module):
#     def __init__(self,dropout):
#         super(MLP, self).__init__()
#         self.linear1 = torch.nn.Linear(100*2, 128)
#         self.linear2 = torch.nn.Linear(128, 64)
#         self.linear3 = torch.nn.Linear(64, 32)
#         self.linear4 = torch.nn.Linear(32, 3)
#         self.dropout = torch.nn.Dropout(DROPOUT) # for linear layer
      
#     def forward(self, x):
#         x = self.dropout(x)      
#         x = torch.nn.functional.relu(self.linear1(x))
#         x = self.dropout(x)
#         x = torch.nn.functional.relu(self.linear2(x))
#         x = self.dropout(x)
#         x = torch.nn.functional.relu(self.linear3(x))
#         x = self.linear4(x)
#         return x


# In[28]:


test_max_f1 = 0
test_max_epoch = 0
Basic_linear_model = MLP(0.2).to(device)
optimizer = torch.optim.Adam(Basic_linear_model.parameters())
loss_fn  = torch.nn.CrossEntropyLoss()
for i in range(50):
    start_time = time.time()
    train_loss,train_acc = training(Basic_linear_model,train_generator,loss_fn,optimizer)
    end_time = time.time()
    train_loss,y_true_train,y_predict_train = evaluation(Basic_linear_model,train_generator,loss_fn)
    dev_loss,y_true_dev,y_predict_dev = evaluation(Basic_linear_model,dev_generator,loss_fn)
    test_loss,y_true,y_predict = evaluation(Basic_linear_model,test_generator,loss_fn)
    print('Epoch: {} [Train] Loss: {:.4f} A: {:.2f} P: {:.2f} R: {:.2f} F1: {:.2f} [Dev] Loss: {:.4f} A: {:.2f} P: {:.2f} R: {:.2f} F1: {:.2f} [Test] Loss: {:.4f} A: {:.2f} P: {:.2f} R: {:.2f} F1: {:.2f}'.format(            
        i+1, 
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
        accuracy_score(y_true,y_predict)*100,
        precision_score(y_true,y_predict,average="macro")*100,
        recall_score(y_true,y_predict,average="macro")*100,
        f1_score(y_true,y_predict,average="macro")*100
    ))
    current_f1 = f1_score(y_true,y_predict,average="macro")*100
    if current_f1>test_max_f1:
        test_max_f1 = current_f1
        test_max_epoch = i+1
    print(test_max_epoch,test_max_f1)


# ### graph SAGE

# In[14]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import itertools
import time

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score


# In[15]:


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


# In[16]:


len(graph_head)


# In[17]:


g = dgl.graph((graph_head, graph_tail), num_nodes=len(e2id))
g.ndata['embedding'] = torch.tensor(entities_emb)

bg = dgl.to_bidirected(g) #bidirection
bg.ndata['embedding'] = torch.tensor(entities_emb)


# In[18]:


def generate_graph(input_df):
    head = []
    tail = []
    y = []
    for index,row in input_df.iterrows():
        citing_paper = row["node1"]
        cited_paper = row["node2"]
        if citing_paper not in e2id or cited_paper not in e2id:
            continue
        citing_paper_id = e2id[citing_paper]
        cited_paper_id = e2id[cited_paper]
        head.append(citing_paper_id)
        tail.append(cited_paper_id)
        y.append(r2id[row["label"]])
    return head,tail,y


# In[19]:


def test_func(epoch_num = 100):
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

                predict_result = pred(dev_g, hidden)
                predict_y = predict_result.argmax(1)
                true_y = dev_y_label
                print('DEV ACC:[{:.2f}]  P:[{:.2f}]  R:[{:.2f}]  F1:[{:.2f}]'.format(accuracy_score(true_y,predict_y)*100,
                precision_score(true_y,predict_y,average="macro")*100,
                recall_score(true_y,predict_y,average="macro")*100,
                f1_score(true_y,predict_y,average="macro")*100))
                # confusion = confusion_matrix(true_y,predict_y)/len(true_y)  
                # print("confusion matrix:\n",confusion)

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


# In[20]:


train_head,train_tail,train_y = generate_graph(sci_train)
test_head,test_tail,test_y = generate_graph(sci_test)
dev_head,dev_tail,dev_y = generate_graph(sci_dev)


# In[21]:


train_y_label = torch.tensor(train_y)
test_y_label = torch.tensor(test_y)
dev_y_label = torch.tensor(dev_y)


# In[22]:


train_g = dgl.graph((train_head, train_tail), num_nodes=g.number_of_nodes())
test_g = dgl.graph((test_head, test_tail), num_nodes=g.number_of_nodes())
dev_g = dgl.graph((dev_head, dev_tail), num_nodes=g.number_of_nodes())


# In[23]:


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

# class MLPPredictor(nn.Module):
#     def __init__(self, h_feats,output_class):
#         super().__init__()
#         self.W1 = nn.Linear(h_feats * 2, 16)
#         self.W2 = nn.Linear(16, output_class)

#     def apply_edges(self, edges):
#         h = torch.cat([edges.src['h'], edges.dst['h']], 1)
#         x = self.W1(h)
#         x = F.relu(x)
#         x = self.W2(x)
#         return {'score': x}

#     def forward(self, g, h):
#         with g.local_scope():
#             g.ndata['h'] = h
#             g.apply_edges(self.apply_edges)
#             return g.edata['score']

# # Use MLP (my previous method)
class MLPPredictor(nn.Module):
    def __init__(self, h_feats,output_class):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, 64)
        self.W2 = nn.Linear(64, 32)
        self.W3 = nn.Linear(32, output_class)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        x = self.W1(h)
        x = F.relu(x)
        x = self.W2(x)
        x = F.relu(x)
        x = self.W3(x)
        return {'score': x}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


# In[26]:


g.ndata['embedding'].dtype


# In[41]:


model = GraphSAGE(g.ndata['embedding'].shape[1], 100)
pred = MLPPredictor(100,3)
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
loss = nn.CrossEntropyLoss()
test_func()


# In[52]:





# In[60]:


# bir
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
model = GraphSAGE(bg.ndata['embedding'].shape[1], 100)
pred = MLPPredictor(100,3)
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
loss = nn.CrossEntropyLoss()
test_func()


# In[61]:


# bir
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
model = GraphSAGE(bg.ndata['embedding'].shape[1], 100)
pred = MLPPredictor(100,3)
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
loss = nn.CrossEntropyLoss()
test_func()


# In[ ]:




