# encoding:utf-8

import codecs
import random
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import pdb, time
from random import shuffle
import math
import copy
from draw import draw_attention
import pickle
import torch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)
def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile1 = open(train_file_path, 'r')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend( [emotion] + clause.split())
    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words)) # 每个词及词的位置
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words)) # 每个词及词的位置
    
    w2v = {}
    inputFile2 = open(embedding_path, 'r')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1) # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend( [list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)] )

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    
    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos
import torch
class Dataset:
    def __init__(self,config,train_set,test_set):
        self.config=config
        self.train_set=copy.deepcopy(train_set)
        self.test_set=copy.deepcopy(test_set)
        self.unk_id=1
        self.word_vocab=self.construct_vocab(train_set+test_set)
        self.labelvocab=self.construct_label_vocab(train_set+test_set)
        self.update_label(self.train_set)
        self.update_label(self.test_set)
        self.vocabsize=len(self.word_vocab)
        self.classsize=len(self.labelvocab)
    def construct_vocab(self,dataset):
        prefix=['<pad>','<unk>']
        count={}
        for sample in dataset:
            for label in sample['inputs']:
                for word in label:
                    if word not in count:
                       count[word]=1
                    else:count[word]+=1
        flatten=[item for item in count.items()]
        sort=sorted(flatten,key=lambda x:x[1],reverse=True)
        cutoff=[item[0] for item in sort][:self.config.maxvocab-2]
        final=prefix+cutoff
        wordvocab={}
        for idx,item in enumerate(final):
            wordvocab[item]=idx
        return wordvocab
    def construct_label_vocab(self,dataset):
        label_dict={'null':0}
        for sample in dataset:
            for label in sample['labels']:
                if label not in label_dict:
                    label_dict[label]=len(label_dict)
        return label_dict
    def update_label(self,dataset):
        for sample in dataset:
            sample['labels']=[self.labelvocab[label] for label in sample['labels']]
    def constrcut_batch(self,batch):
        maxdoc=max([sample['doc_len'] for sample in batch])
        maxword=max([len(x) for sample in batch for x in sample['inputs']])
        batchsize=len(batch)
        inputs,contextlen,sentencelen,labels,adjs=\
            np.zeros([batchsize,maxdoc,maxword],dtype=np.int32),\
            np.ones([batchsize],dtype=np.int32),\
            np.ones([batchsize,maxdoc],dtype=np.int32),\
            np.zeros([batchsize,maxdoc],dtype=np.int32),\
            np.zeros([batchsize,maxdoc,maxdoc],dtype=np.float32)

            
#        inputs,contextlen,sentencelen,labels,adjs=\
#            torch.zeros(batchsize,maxdoc,maxword).long().cuda(),\
#            torch.ones(batchsize).long().cuda(),\
#            torch.ones(batchsize,maxdoc).long().cuda(),\
#            torch.zeros(batchsize,maxdoc).long().cuda(),\
#            torch.zeros(batchsize,maxdoc,maxdoc).float().cuda()
        for idx,sample in enumerate(batch):
            for idj,sentence in enumerate(sample['inputs']):
                inputs[idx][idj][:len(sentence)]=[self.word_vocab.get(word,self.unk_id) for word in sentence]
                sentencelen[idx][idj]=len(sentence)
            contextlen[idx]=sample['doc_len']
            labels[idx][:sample['doc_len']]=sample['labels']
            for pair in sample['pairs']:
                adjs[idx][pair[0]-1,pair[1]-1]=1
        inputs,contextlen,sentencelen,labels,adjs=\
            torch.Tensor(inputs).long().cuda(),\
            torch.Tensor(contextlen).long().cuda(),\
            torch.Tensor(sentencelen).long().cuda(),\
            torch.Tensor(labels).long().cuda(),\
            torch.Tensor(adjs).float().cuda()
        # print(inputs,contextlen,sentencelen,labels,adjs)
        # a=input()
        return inputs,contextlen,sentencelen,labels,adjs
    def train_iter(self):
        shuffle(self.train_set)
        num=math.ceil(len(self.train_set)/self.config.batch_size)
        for i in range(num):
            yield self.train_set[i*self.config.batch_size:(i+1)*self.config.batch_size],self.constrcut_batch(self.train_set[i*self.config.batch_size:(i+1)*self.config.batch_size])
    def test_iter(self):
        num=math.ceil(len(self.test_set)/self.config.batch_size)
        for i in range(num):
            yield self.test_set[i*self.config.batch_size:(i+1)*self.config.batch_size],self.constrcut_batch(self.test_set[i*self.config.batch_size:(i+1)*self.config.batch_size])   
def load_data(input_file, maxdoc=30, maxword=30):
    print('load data_file: {}'.format(input_file))

    y_pairs=[]
    doc_id = []
    doc_len=[]
    inputFile = open(input_file, 'r',encoding='utf-8')
    xs=[]
    labelss=[]
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        doc_len.append(d_len)
        y_pairs.append(pairs)
        pos, cause = zip(*pairs)
        labels=[]
        x=[]
        for i in range(d_len):
            source=inputFile.readline().strip().split(',')
            words = source[-1].split()
            label= source[1]
            labels.append(label)
            x.append(words[:maxword])
        xs.append(x)
        labelss.append(labels)
    datas=[]
    for idx,item in enumerate(doc_len):
        if item>maxdoc:
            continue
        data={}
        data['doc_id']=doc_id[idx]
        data['doc_len']=doc_len[idx]
        data['pairs']=y_pairs[idx]
        data['inputs']=xs[idx]
        data['labels']=labelss[idx]
        datas.append(data)
        # print(data)
        # a=input()
    print('load data done!\n')
    print('length'+str(len(datas)))
    print('len1'+str(sum([data['doc_len'] for data in datas])))
    return datas

def acc_prf(pred_y, true_y, doc_len, average='binary'): 
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, p, r, f1
def adjust_learning_rate(optimizer):
    cur_lr = optimizer.param_groups[0]['lr']
    # adj_lr = cur_lr / 2
    adj_lr = cur_lr * 0.5
    print("Adjust lr to ", adj_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = adj_lr
class Config(object):
      batch_size=16
      maxvocab=24000
      clip_grad=5
      l2=1e-5
      lr=0.001
      embsize=200
      hidden_size=128
      drop_rate=0.5
      bi=True
      pos_num=200
      epoch=400
      selfatt=True
      selfmulatt=True
      bio_first=False
      model_name='att'
      onlytest=False
def process_pair(pred_pair,golden_pair):
    pairs=[]
    for i in range(pred_pair.shape[0]):
        for j in range(pred_pair.shape[1]):
            if pred_pair[i,j]>0.5:
                pairs.append((i+1,j+1))
    num=0
    for item in pairs:
        if item in golden_pair:
            num+=1
    return num,len(pairs),len(golden_pair)
def process(pred_pair):
    pairs=[]
    for i in range(pred_pair.shape[0]):
        for j in range(pred_pair.shape[1]):
            if pred_pair[i,j]>0.5:
                pairs.append((i+1,j+1))
    return pairs
def evaluates(model,test_iter,show=False):
    model.eval()
    label_vocab=dict([(v,u) for u,v in model.labelvocab.items()])
    pred=[]
    golden=[]
    pred1=[]
    golden1=[]
    n1s=0
    n2s=0
    n3s=0
    em=0
    nums=0
    errorr_esults=[]
    for source,batch in test_iter:
        # logits_for_class,logits_for_pairs,att1,att2=model.test(batch)#b,c;b,c,c
        logits_for_emo, logits_for_cause,emo,cause= model.process(batch)  # b,c;b,c,c
        logits_for_emo=(torch.sigmoid(logits_for_emo)>0.5).long()
        logits_for_cause = (torch.sigmoid(logits_for_cause)>0.5).long()
        for idx,item in enumerate(source):
            cur_length=item['doc_len']
            pred.extend(logits_for_emo[idx][:cur_length].tolist())
            golden.extend(emo[idx][:cur_length].tolist())
            pred1.extend(logits_for_cause[idx][:cur_length].tolist())
            golden1.extend(cause[idx][:cur_length].tolist())
            assert(len(pred)==len(golden))
#             pred_pair=logits_for_pairs[idx][:cur_length,:cur_length]
#             golden_pair=item['pairs']
# #            if(show):
# #                print(pred_pair)
# #                print(golden_pair)
# #                a=input('hahah')
#             n1,n2,n3=process_pair(pred_pair,golden_pair)
#             n1s+=n1
#             n2s+=n2
#             n3s+=n3
#             nums+=1
    p = precision_score(golden, pred, average='binary')
    r = recall_score(golden, pred, average='binary')
    f1 = f1_score(golden, pred, average='binary')
    p1 = precision_score(golden1, pred1, average='binary')
    r1= recall_score(golden1, pred1, average='binary')
    f11= f1_score(golden1, pred1, average='binary')
    # p=0
    # r=0
    # f1=0
    # n2s+=0.00000001
    # n3s+=0.00000001
    # print('class#'+str(p)+'#'+str(r)+'#'+str(f1))
    # print('pair#'+str(n1s/n2s)+'#'+str(n1s/n3s)+'#'+str(2*n1s/(n2s+n3s)))
    # print('em#'+str(em/nums))
    # print('sdsdsdsd')
    print([None,p,r,f1,p1,r1,f11])
    return [None,p,r,f1,p1,r1,f11]
def evaluate(model,test_iter,ecmodel,show=False):
    model.eval()
    label_vocab=dict([(v,u) for u,v in model.labelvocab.items()])
    pred=[]
    golden=[]
    n1s=0
    n2s=0
    n3s=0
    em=0
    nums=0
    errorr_esults=[]
    for source,batch in test_iter:
        # logits_for_class,logits_for_pairs,att1,att2=model.test(batch)#b,c;b,c,c
        logits_for_pairs, logits_for_class= model.process(batch,ecmodel)  # b,c;b,c,c
        att1=model.att_value
        fused=model.fused
        logits_for_pairs=torch.sigmoid(logits_for_pairs)
        for idx,item in enumerate(source):
            cur_length=item['doc_len']
            pred.extend(logits_for_class[idx][:cur_length].tolist())
            golden.extend(item['labels'])
            assert(len(pred)==len(golden))
            pred_pair=logits_for_pairs[idx][:cur_length,:cur_length]
            golden_pair=item['pairs']
#            if(show):
#                print(pred_pair)
#                print(golden_pair)
#                a=input('hahah')
            n1,n2,n3=process_pair(pred_pair,golden_pair)
            n1s+=n1
            n2s+=n2
            n3s+=n3
            if n1==n2 and n2==n3:
                em+=1
                if(show):
                    tmp2=att1[idx].cpu().numpy()
                    tmp3 = fused[idx].cpu().numpy()
                    # tmp2=att2[idx]
                    for idj in range(cur_length):
                        # if item['labels'][idj]!=0:
                            # print(label_vocab[item['labels'][idj]])
                            # print(item['inputs'][idj])
                            # draw_attention([label_vocab[item['labels'][idj]]],item['inputs'][idj],tmp1[idj:idj+1][:,:len(item['inputs'][idj])],str(nums)+'_0')
#                            a=input('haha')
                            kl=1
                            for idz in range(cur_length):
                                if pred_pair[idj,idz]>0.5:
                                    adjs = tmp2[idj, idz]
                                    adjs1 = tmp3[idj, idz]
                                    print(adjs[:len(item['inputs'][idj]),:len(item['inputs'][idz])])
                                    print(adjs1[:len(item['inputs'][idj]),:len(item['inputs'][idz])])
                                    draw_attention(item['inputs'][idj],item['inputs'][idz],adjs[:len(item['inputs'][idj]),:len(item['inputs'][idz])],str(nums)+'_'+str(kl))
                                    # a=input('haha')
                                    kl+=1
            else:
                content=item['inputs']
                # truelabel=[label_vocab[label] for label in item['labels']]
                # predlabel=[label_vocab[label] for label in logits_for_class[idx][:cur_length].tolist()]
                truetuple=item['pairs']
                predtuple=process(pred_pair)
                errorr_esults.append([content,truetuple,predtuple])
            nums+=1
    # p = precision_score(golden, pred, average='macro')
    # r = recall_score(golden, pred, average='macro')
    # f1 = f1_score(golden, pred, average='macro')
    p=0
    r=0
    f1=0
    n2s+=0.00000001
    n3s+=0.00000001
    print('class#'+str(p)+'#'+str(r)+'#'+str(f1)) 
    print('pair#'+str(n1s/n2s)+'#'+str(n1s/n3s)+'#'+str(2*n1s/(n2s+n3s)))  
    print('em#'+str(em/nums))
    print('sdsdsdsd')
    return [em/nums,p,r,f1,n1s/n2s,n1s/n3s,2*n1s/(n2s+n3s)]
    # return [em/nums,p,r,f1,n1s/n2s,n1s/n3s,2*n1s/(n2s+n3s)],errorr_esults
#    pass
from models import joint_model,ec_model
#import time
if __name__ == '__main__':  
    torch.cuda.set_device(0)    
    config=Config()
    data=load_data('./data_combine/fold2_train.txt')
    data1=load_data('./data_combine/fold2_test.txt')
    print(len(data)+len(data1))
    print(sum([item['doc_len'] for item in data])+sum([item['doc_len'] for item in data1]))
    dataset=Dataset(config,data,data1)
    config.tag_num=dataset.classsize
    config.labelvocab=dataset.labelvocab
    print(config.labelvocab)
    config.vocab_size=dataset.vocabsize
    print(config.vocab_size)
    config.vocab=dataset.word_vocab
    print(type(dataset))
    # print(config.vocab_size)
    # a=input()
    # tmp=[]
    # for datas,datas1 in dataset.train_iter():
    #     tmp.append([datas,[tmp.cpu().numpy() for tmp in datas1]])
    ecmodel = ec_model(config)
    ecmodel.eval()
    ecmodel.cuda()
    ecmodel.load_state_dict(torch.load('./model_save/modelec_'+config.model_name+'.pth'))
    config.drop_rate=0.3
    config.onlytest=False
    model=joint_model(config)
    # model=ecmodel
    model.load_emd('./data_combine/w2v_200.txt')
    if config.onlytest:
     model.load_state_dict(torch.load('./model_save/model_'+config.model_name+'.pth'))
    model.cuda()
    optimizer=torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    count=0
    bestacc=None
    early_stop=0
    bestmodel=None
    records=[]
    if config.onlytest:
        config.epoch=1
    for idx in range(config.epoch):
        if not config.onlytest:
            start=time.time()
            model.train()
            losses=0
            num=0
            for _,batch in dataset.train_iter():
                optimizer.zero_grad()
                loss=model(batch,ecmodel)
                # loss = model(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), config.clip_grad)
                optimizer.step()
                losses+=loss.cpu().item()
                num+=1
                loss=None
    #        a=input('haha')
            print('train_loss:'+str(idx)+':'+str(losses/num))
            print('traintime:'+str(time.time()-start))
           # if idx>=20:evaluate(model,dataset.train_iter(),False)
            with torch.no_grad():
             # if idx >= 5: evaluate(model, dataset.train_iter(), False)
             acc=evaluate(model,dataset.test_iter(),ecmodel)
             # acc = evaluates(model, dataset.test_iter())
             records.append(acc)
            print('best:')
            print(bestacc)
    #         if idx>=150:
    #          acc=evaluate(model,dataset.test_iter(),True)
            if bestacc is None:
                bestacc=acc
            if bestacc[6]<=acc[6]:
                bestacc=acc
                count=0
                del bestmodel
                bestmodel=copy.deepcopy(model)
            else:
                count+=1
                if count>15:
                    early_stop+=1
                    count=0
                    if early_stop>3:
                        break
                    else:
                        model.load_state_dict(bestmodel.state_dict())
                        adjust_learning_rate(optimizer)
            torch.save(bestmodel.state_dict(),'./model_save/modelec_'+config.model_name+'.pth')
        else:
            with torch.no_grad():
             _,haha=evaluate(model,dataset.test_iter(),None)
    print('sdsdsdsd')
    with open('resultss.txt','w',encoding='utf-8') as file:
        for item in haha:
            for i in range(len(item[0])):
                file.write(str(i+1)+' '+str(' '.join(item[0][i]))+' '+'\n')
            file.write(str(item[1])+'\n')
            file.write(str(item[2])+'\n')
            file.write('#'+'\n')
    file.close()
#             evaluate(model,dataset.test_iter(),True)
#    with open('results.pkl','wb') as file:
#        pickle.dump(records,file)         

#            count=0
#            del bestmodel
#            bestmodel=copy.deepcopy(model)
#        else:
#            count+=1
#            if count>20:
#                early_stop+=1
#                count=0
#                if early_stop>3:
#                    break
#                else:
#                    model.load_state_dict(bestmodel.state_dict())
#                    adjust_learning_rate(optimizer)
#    print('best'+str(bestacc))
#    if f1 > best_per:
#        early_counter = 0
#        best_per = f1
#        del best_model
#        best_model = copy.deepcopy(ner_model)
#    else:
#        early_counter += 1
#        if early_counter > config.lr_patience:
#            decay_counter += 1
#            early_counter = 0
#            if decay_counter > config.decay_patience:
#                break
#            else:
#                ner_model.load_state_dict(best_model.state_dict())
#                adjust_learning_rate(optimizer)            
    
    
    
