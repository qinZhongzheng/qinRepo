
from utils import data_load as dal


#Count the numbers in Bert words
def count_words(model,pre_dict,g):
    dict = {}
    pre = {}
    list_con = []
    for id in pre_dict.keys() :
        temp_dict = pre_dict[id]
        id_true =temp_dict['true']
        id_f = temp_dict['false']
        in_put_true = g.ndata['input_ids'][id_true]
        in_put_false = g.ndata['input_ids'][id_f]
        for i in in_put_true:
            text_true = model.tokenizer.decode(i)
            list_con.append(text_true)
        pre['true'] = list_con
        list_con=[]
        for i in in_put_false:
            text_false = model.tokenizer.decode(i)
            list_con.append(text_false)
        pre['false'] = list_con
        list_con=[]
        dict[id] = pre
    return dict

def not_at(dict,dataset):
    voacb = []
    temp_list = []
    temp_word = []
    for id in dict.keys():
        true_list = dict[id]['true']
        false_list = dict[id]['false']
        for line in false_list :
            s1 = line.find("<s>")
            s2 = line.find("</s>")
            linesub = line[(s1+3):s2]
            temp_list.append(linesub)
        for v in temp_list:
            temp = v.split(" ")
            for l in temp:
                temp_word.append(l)
        for l in temp_word:
            for line in true_list:
                if line.find(l) == -1:
                    voacb.append(l)
        temp_list =[]
        temp_word = []
        voacb = set(voacb) 
        f = open('data/class/' + dataset +'_'+id+'.notv.txt', 'w',encoding='utf-8')
        f.write('\n'.join(voacb))
        f.close()

def all_word(file_name,dataset):
    word_dic_freq = {}
    appeared = set()
    line_list = []
    # for key in dict.keys():
    #     all_list = []    
    #     true_list = dict[key]['true']
    #     false_list = dict[key]['false']
    #     all_list = true_list+false_list
    with open(file_name) as f :
        lines = f.readlines()
        for line in lines:
            line_list = line.split()
            for word in line_list:
                if word in appeared:
                    word_dic_freq[word] += 1
                else:
                    word_dic_freq[word] = 1
                    appeared.add(word)
    dal.to_json(dataset,"word_true_dic",word_dic_freq)

def data_dict(dataset,g):
    dirc = {}
    pre = {}
    fileName = dataset+'_45.json'
    data = dal.to_data(fileName)
    label_list = g.ndata['label']
    labels = set(label_list)
    for label in labels :
        list_con =[]
        for i in data['true_id']:
            if label_list[i] == label.item():
                list_con.append(i)
        pre['true'] = list_con
        list_con =[]
        for i in data['false_id']:
            if label_list[i] == label.item():
                list_con.append(i)
        pre['false'] = list_con
        dirc[str(label.item())] = pre 
    return dirc

#print the loss
def Print_loss(loss,loss_list):
    loss_list.append(str(loss.item()))
    return loss_list

if __name__ == '__main__':
    file_name = 'data/corpus/ohsumed.clean.txt'
    all_word(file_name,"ohsumed")