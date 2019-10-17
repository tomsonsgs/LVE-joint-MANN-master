# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:39:05 2019

@author: tomson
"""
from prepare_data import *
from models import joint_model
import torch
import time
import copy
#import time
if __name__ == '__main__':  
    torch.cuda.set_device(0)    
    config=Config()
    data=load_data('./data_combine/fold2_train.txt')
    data1=load_data('./data_combine/fold2_test.txt')
    dataset=Dataset(config,data,data1)
    config.tag_num=dataset.classsize
    config.labelvocab=dataset.labelvocab
    config.vocab_size=dataset.vocabsize
    config.vocab=dataset.word_vocab
    print(type(dataset))
    print(type(Dataset))
    tmp=[]
    for datas,datas1 in dataset.train_iter():
        tmp.append([datas,[tmp.cpu().numpy() for tmp in datas1]])
    model=joint_model(config)
#    model.load_emd('./data_combine/w2v_200.txt')
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
                loss=model(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), config.clip_grad)
                optimizer.step()
                losses+=loss.cpu().item()
                num+=1
                loss=None
    #        a=input('haha')
            print('train_loss:'+str(idx)+':'+str(losses/num))
            print('traintime:'+str(time.time()-start))
    #        if idx>=20:evaluate(model,dataset.train_iter(),False)
            with torch.no_grad():
             acc=evaluate(model,dataset.test_iter())
             records.append(acc)
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
                if count>30:
                    early_stop+=1
                    count=0
                    if early_stop>3:
                        break
                    else:
                        model.load_state_dict(bestmodel.state_dict())
                        adjust_learning_rate(optimizer)
            torch.save(bestmodel.state_dict(),'./model_save/model_'+config.model_name+'.pth')
        else:            
            with torch.no_grad():
             _,haha=evaluate(model,dataset.test_iter(),False)
    with open('resultss.txt','w',encoding='utf-8') as file:
        for item in haha:
            for i in range(len(item[0])):
                file.write(str(i+1)+' '+str(' '.join(item[0][i]))+' '+str(item[1][i])+' '+str(item[2][i])+'\n')
            file.write(str(item[3])+'\n')
            file.write(str(item[4])+'\n')
            file.write('#'+'\n')
    file.close()