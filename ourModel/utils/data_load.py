
from utils import *
import torch as th
import json
import random
# gpu = th.device('cuda:0')
cpu = th.device('cpu')
gpu = th.device('cpu')
import csv
import utils as ul

def total_data_class(dataset):
    train_dic = {}
    doc_name_list = []
    doc_test_list = []
    doc_train_list = []
    labels = []
    with open('data/' + dataset + '.txt', 'r') as f:
        lines = f.readlines()
        for line in lines :
            doc_name_list.append(line.strip())
            temp = line.split("\t")
            if temp[1].find('test') != -1:
                doc_test_list.append(line.strip())
                label = temp[2].strip()
                if label not in labels:
                    labels.append(label)
            elif temp[1].find('train') != -1:
                label = temp[2].strip()
                if label not in labels:
                    labels.append(label)
                doc_train_list.append(line.strip())
    for label in labels:
        temp1 = []
        for i in doc_train_list :
            if label in i :
                temp1.append(doc_name_list.index(i))
        train_dic[label] = temp1

        # for i in doc_test_list :
        #     if label in i :
        #         temp2.append(doc_name_list.index(i))
        # test_dic[label] = temp2
        
    return train_dic,labels

def to_loss_1(y_true,y_pred_label_test,cls_pred,cls_test):
    yy = th.zeros(23, 768, device=gpu)
    yyyy = [0.0000001] * 23
    # yyyy = th.tensor(yyyy)
    for i,l in enumerate(y_true):
        ci = cls_pred[i] #
        yy[l] = yy[l] + ci
        yyyy[l] +=1
    for i in range(23):
        yy[i] /= yyyy[i]
   
    xx = th.zeros(23, 768, device=gpu)
    xxxx = [0.0000001] * 23
    # xxxx = th.tensor(xxxx)
    for i,l in enumerate(y_pred_label_test):
        ci = cls_test[i]
        xx[l] = xx[l] + ci
        xxxx[l] +=1
    for i in range(23):
        xx[i] /= xxxx[i]
    j = 0
    loss_1 = None
    for i, y in enumerate(yy):
        x = xx[i]
        if y.sum(dim=0) != 0 and x.sum(dim=0) != 0:
            l = th.cosine_similarity(x,y,dim=0)
            j=j+1
            if loss_1:
                loss_1 = loss_1 + l   
            else:
                loss_1 = l
    if loss_1:
        loss_1 = loss_1 / j

    return loss_1

def to_loss(train_data,test_data):
    train_matrix =th.zeros(23, 768, device=gpu)
    tr_mean = [0.0000001] * 23
    test_matrix = th.zeros(23, 768, device=gpu)
    te_mean = [0.0000001] * 23
    for i ,data in enumerate(train_data):
        ci = th.tensor(data['cls'],device=gpu)
        train_label = int(data['true_label'])
        train_matrix[train_label] = ci + train_matrix[train_label]
        tr_mean[train_label]+=1
    for i in range(23):
        train_matrix[i] /= tr_mean[i]
    for data in test_data:
        ci = th.tensor(data['cls'],device=gpu)
        test_id = int(data['true_label'])
        test_matrix[test_id] = ci + test_matrix[test_id]
        te_mean[test_id]+=1
    for i in range(23):
        test_matrix[i] /= te_mean[i]
    j = 0
    loss_1 = None
    for i, y in enumerate(train_matrix):
        x = test_matrix[i]
        if y.sum(dim=0) != 0 and x.sum(dim=0) != 0:
            l = th.cosine_similarity(x,y,dim=0)
            j=j+1
            if loss_1:
                loss_1 = loss_1 + l   
            else:
                loss_1 = l
    if loss_1:
        loss_1 = loss_1 / j

    return loss_1

def list_line(fileName):
    context = []
    with open (fileName,'r') as f:
        lines = f.readlines()
        for line in lines:
            context.append(line.strip())
    return context

def to_data(filename):
    with open(filename,'r',encoding="utf-8") as f :
        data = json.load(f)
    return data
   
def to_json(dataset,name,data):
    f = open('data/' + dataset + '_' + name + '.json', 'w',encoding="utf-8")
    json.dump(data,f)
    f.close()

def to_dict(dataset,test_error_id,test_true_l,test_error_l,cls):
    doc_content_list=[]
    dict = []
    cont = {}
    temp = {}
    with open('data/corpus/' + dataset + '_shuffle.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc_content_list.append(line.strip())
    for i,id in enumerate(test_error_id):
        cont["id"] = float(test_error_id[i])
        cont["text"] = doc_content_list[id]
        cont["true_label"] = float(test_true_l[i])
        cont["pred_label"] = float(test_error_l[i])
        cont["cls"] = cls[i]
        temp = cont.copy()
        dict.append(temp)
    return dict

def to_masg(data,dataset):
    with open("data/corpus/"+dataset+"_labels.txt",'r',encoding="utf-8") as f:
        lines = f.readlines()
    doc_label =[]
    for line in lines:
            doc_label.append(line.strip())
    list=[0]*23
    for d in data:
        list[int(d["true_label"])] +=1 
    dict = {}
    for i,l in enumerate(doc_label):
        dict[l] = list[i]

    return dict
   
def to_readData(dataset):
    label_list =[]
    context_list =[]
    context_clean_list =[]
    with open("data/IMDB/test.tsv",encoding='utf-8') as f:
        tsvreader = csv.reader(f,delimiter='\t')
        for line in tsvreader:
            label_list.append(line[0])
            context_list.append(line[1])
            context_clean_list.append(ul.clean_str(line[1]))
    shuffle_doc_context_str = '\n'.join(context_clean_list)
    f = open('data/corpus/' + dataset + '.clean.txt', 'w',encoding='utf-8')
    f.write(shuffle_doc_context_str)
    f.close()

def min_label(dataset):
    doc_name_list =[]
    train_list=[]
    with open('data/' + dataset + '.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc_name_list.append(line.strip())
    # for i,line in enumerate(doc_name_list):
    #     temp = line.split("\t")
    #     if temp[1].find('train') != -1:
    #             doc_train_id.append(i)
    #     label_set.add(temp[2])
    # label_list = list(label_set)

    train_dic,labels = total_data_class(dataset)
    for l in labels:
        i = 0
        for id in train_dic[l]:
            i+=1
            if i>len(train_dic[l])/2:
                temp = doc_name_list[id].split("\t")
                doc_name_list[id] = doc_name_list[id].replace(temp[1],"test")
    train_list_str = '\n'.join(doc_name_list)
    f = open('data/' + dataset + '_new.txt', 'w',encoding='utf-8')
    f.write(train_list_str)
    f.close()

def readoftrain(dataset):
    doc_name_list =[]
    context_list=[]
    doc_name_new_list =[]
    train_context_list_new_id=[]
    train_context_list_new =[]
    with open('data/' + dataset + '.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            doc_name_list.append(line.strip())
    with open('data/corpus/'+dataset+'.clean.txt','r') as f:
        lines_t = f.readlines()
        for lines in lines_t:
            context_list.append(lines)
    train_dic,labels = total_data_class(dataset)
    for key in train_dic.keys():
        random.shuffle(train_dic[key])
    for l in labels:
        i = 0
        for id in train_dic[l]:
            i+=1
            if i<=len(train_dic[l])*1/2:
                # temp = doc_name_list[id].split("\t")
                # doc_name_list[id] = doc_name_list[id].replace(temp[1],"test")
                # doc_name_new_list.append(doc_name_list[id])
                train_context_list_new_id.append(id)
    for i,d in enumerate(context_list):
        if i not in train_context_list_new_id:
            train_context_list_new.append(d.strip())
    for i,d in enumerate(doc_name_list):
        if i not in train_context_list_new_id:
            doc_name_new_list.append(d)
    train_list_str = '\n'.join(doc_name_new_list)
    con_str = '\n'.join(train_context_list_new)
    f = open('data/' + dataset + '1_new.txt', 'w',encoding='utf-8')
    f.write(train_list_str)
    f.close()

    f = open('data/corpus/' + dataset + 'new.clean.txt', 'w',encoding='utf-8')
    f.write(con_str)
    f.close()

def to_setcalculate(set1 ,set2):
    return set(set1)&set(set2)
            
            






     
        

if __name__ == '__main__':
    dataName = 'R52'
    readoftrain(dataName)
    # with open("ohsumed_35.json") as f :
    #     set1 = json.load(f)
    # with open("ohsumed_40.json") as f :
    #     set2 = json.load(f)     
    # print("set1: {}".format(len(set1["true_id"])))
    # print("set2: {}".format(len(set2["true_id"])))
    # set = to_setcalculate(set1["true_id"],set2["true_id"])
    # print("set: {}".format(len(set)))
    # Word_frequcency_stat('data/corpus/ohsumed_vocab.txt')  
    # lisr = list_line('data/corpus/ohsumed_vocab.txt')
    # print(lisr)     




 








