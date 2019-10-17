# -*- coding: utf-8 -*-

import torch
from torch import nn
#from pytorch_crf import CRF
import copy
import numpy as np
from encoder import EncoderRNN,ContextRNN,selfatt,mulatt
def length2mask(length,maxlength):
    size=list(length.size())
    length=length.unsqueeze(-1).repeat(*([1]*len(size)+[maxlength])).long()
    ran=torch.arange(maxlength).cuda()
    ran=ran.expand_as(length)
    mask=ran<length
    return mask.float().cuda()
class joint_model(torch.nn.Module):
    def __init__(self,config):
        super(joint_model,self).__init__()
        self.config=config
        self.vocab=config.vocab
        self.labelvocab=config.labelvocab
        self.drop_layer=nn.Dropout(config.drop_rate)
        self.token_lstm=EncoderRNN(config.vocab_size,config.embsize,config.hidden_size)
        if config.selfatt:
            self.context_lstm=ContextRNN(config.hidden_size*(int(config.bi)+1+1+2),config.hidden_size)
        else:self.context_lstm=ContextRNN(config.hidden_size*(int(config.bi)+1+1),config.hidden_size)
        self.hid_linear=torch.nn.Linear(config.hidden_size*(int(config.bi)+1),config.hidden_size)
        self.selfatt=selfatt(config.hidden_size*(int(config.bi)+1))
        self.mulatt=mulatt(config.hidden_size*(int(config.bi)+1))
        self.out_linear=torch.nn.Linear(config.hidden_size,config.tag_num)
        self.pair2=torch.nn.Linear(config.hidden_size,1)
        if config.bio_first:
            self.pair1=torch.nn.Linear(config.hidden_size*4,config.hidden_size)
        else:self.pair1=torch.nn.Linear(config.hidden_size*(2+2),config.hidden_size)
#        self.embedding=nn.Embedding(config.vocab_size,config.embsize)
        self.pos_emb=nn.Embedding(config.pos_num,config.hidden_size)
#        self.crf=CRF(config.tag_num,batch_first=True)
        self.classloss=torch.nn.CrossEntropyLoss(reduction='none')
        self.binloss=torch.nn.BCEWithLogitsLoss(reduction='none')
        self.tag_emb=nn.Embedding(config.tag_num,config.hidden_size)
        self.maps=torch.nn.Linear(config.hidden_size*4,config.hidden_size*2)
#        self.pair2
    def flatten(self,inputs):
        inputs=inputs.view([inputs.size(0)*inputs.size(1)]+list(inputs.size())[2:])
        return inputs
    def load_emd(self,embedding_path):
        w2v = {}
        embedding_dim=200
        inputFile2 = open(embedding_path, 'r',encoding='utf-8')
        inputFile2.readline()
        for line in inputFile2.readlines():
            line = line.strip().split(' ')
            w, ebd = line[0], line[1:]
            w2v[w] = ebd
    
        embedding = np.zeros([len(self.vocab),embedding_dim])
        hit = 0
        for item in self.vocab:
            if item in w2v:
                vec = list(map(float, w2v[item]))
                embedding[self.vocab[item]]=vec
                hit += 1
            else:
                vec = list(np.random.rand(embedding_dim) / 5. - 0.1) # 从均匀分布[-0.1,0.1]中随机取
                embedding[self.vocab[item]]=vec
        print('hit'+str(hit))
        self.token_lstm.embedding.weight.data.copy_(torch.Tensor(embedding).float().cuda())
        
    def forward(self,batch):
        inputs,context_len,sentence_len,labels,adjs=batch#b,c,s;b;b,c
        batch_size=inputs.size(0)
        context_size=inputs.size(1)
        word_size=inputs.size(2)
        outputs,(sentences,_)=self.token_lstm(self.flatten(inputs),self.flatten(sentence_len))#bc,s,2h;2,bc,h
        outputsori=outputs
        outputs=outputs.view(batch_size,context_size,word_size,-1)#b,c,s,2h
        sentences=sentences.transpose(0,1).contiguous()
        sentencesori=sentences.view(batch_size*context_size,-1)
        out,att1=self.selfatt(outputsori,sentencesori,self.flatten(length2mask(sentence_len,word_size)))#bc,2h
        out=out.reshape(batch_size,context_size,-1)
        sentences=sentences.view(batch_size,context_size,-1)#b,c,2h
        sentences1=sentences.unsqueeze(-2).repeat(1,1,word_size,1)
        pos_emb=self.pos_emb(torch.arange(context_size).cuda()).unsqueeze(0).repeat(batch_size,1,1)#b,c,h
        if self.config.selfatt:
            sentences=torch.cat([sentences,out,pos_emb],-1)#b,c,5h
        else:sentences=torch.cat([sentences,pos_emb],-1)#b,c,5h
        sentences=self.drop_layer(sentences)
#        print(sentences.size())
        mid_outputs,(context,_)=self.context_lstm(sentences,context_len)#b,c,2h;2,b,h
        mid_outputs=torch.tanh(self.hid_linear(mid_outputs))
        logits_for_class=self.out_linear(mid_outputs)#b,c,tag
        tags=self.tag_emb(torch.argmax(logits_for_class,-1))#b,c,h
        tags1=tags.unsqueeze(1).repeat(1,context_size,1,1)#b,c,c,h
        tags2=tags.unsqueeze(2).repeat(1,1,context_size,1)#b,c,c,h
        loss1=self.classloss(logits_for_class.view(-1,self.config.tag_num),labels.view(-1))*length2mask(context_len,context_size).view(-1)
        loss1=loss1.view(batch_size,-1).sum(-1).mean()
        mid_outputs1=mid_outputs.unsqueeze(1).repeat(1,context_size,1,1)#b,c,c,h
        mid_outputs2=mid_outputs.unsqueeze(2).repeat(1,1,context_size,1)#b,c,c,h
#        outputs=torch.tanh(self.maps(torch.cat([outputs,sentences1],-1)))
        extra,att2=self.mulatt(outputs,length2mask(sentence_len,word_size),att1.reshape(batch_size,context_size,-1))
#        print(extra.size())
        if self.config.bio_first:
            fused=torch.cat([mid_outputs1,mid_outputs2,tags1,tags2],-1)#b,c,c,2h
        else:fused=torch.cat([mid_outputs1,mid_outputs2,extra],-1)#b,c,c,2h
        fused=self.drop_layer(fused)
        logits_for_pairs=self.pair2(torch.tanh(self.pair1(fused))).squeeze(-1)
        context_mask=length2mask(context_len,context_size)
        context_mask=context_mask.unsqueeze(1)*context_mask.unsqueeze(2)
#        tadjs=copy.deepcopy(adjs.data)
#        tadjs[tadjs==1]=100
#        tadjs[tadjs==0]=1
        loss2=self.binloss(logits_for_pairs,adjs)*context_mask
        loss2=loss2.sum(-1).sum(-1).mean()
        return loss1+0.5*loss2
#        return loss2
    def test(self,batch):
        inputs,context_len,sentence_len,labels,adjs=batch#b,c,s;b;b,c
        batch_size=inputs.size(0)
        context_size=inputs.size(1)
        word_size=inputs.size(2)
        outputs,(sentences,_)=self.token_lstm(self.flatten(inputs),self.flatten(sentence_len))#bc,s,2h;2,bc,h
        outputsori=outputs
        outputs=outputs.view(batch_size,context_size,word_size,-1)#b,c,s,2h
        sentences=sentences.transpose(0,1).contiguous()
        sentencesori=sentences.view(batch_size*context_size,-1)
        out,att1=self.selfatt(outputsori,sentencesori,self.flatten(length2mask(sentence_len,word_size)))#bc,2h
        out=out.reshape(batch_size,context_size,-1)
        sentences=sentences.view(batch_size,context_size,-1)#b,c,2h
        sentences1=sentences.unsqueeze(-2).repeat(1,1,word_size,1)
        pos_emb=self.pos_emb(torch.arange(context_size).cuda()).unsqueeze(0).repeat(batch_size,1,1)#b,c,h
        if self.config.selfatt:
            sentences=torch.cat([sentences,out,pos_emb],-1)#b,c,5h
        else:sentences=torch.cat([sentences,pos_emb],-1)#b,c,5h
        sentences=self.drop_layer(sentences)
#        print(sentences.size())
        mid_outputs,(context,_)=self.context_lstm(sentences,context_len)#b,c,2h;2,b,h
        mid_outputs=torch.tanh(self.hid_linear(mid_outputs))
        logits_for_class=self.out_linear(mid_outputs)#b,c,tag
        tags=self.tag_emb(torch.argmax(logits_for_class,-1))#b,c,h
        tags1=tags.unsqueeze(1).repeat(1,context_size,1,1)#b,c,c,h
        tags2=tags.unsqueeze(2).repeat(1,1,context_size,1)#b,c,c,h
        logits_for_class=torch.argmax(logits_for_class,-1)#b,c
#        loss1=self.classloss(logits_for_class.view(-1,self.config.tag_num),labels.view(-1))*length2mask(context_len,context_size).view(-1)
#        loss1=loss1.view(batch_size,-1).sum(-1).mean()
        mid_outputs1=mid_outputs.unsqueeze(1).repeat(1,context_size,1,1)#b,c,c,h
        mid_outputs2=mid_outputs.unsqueeze(2).repeat(1,1,context_size,1)#b,c,c,h
#        outputs=torch.tanh(self.maps(torch.cat([outputs,sentences1],-1)))
        extra,att2=self.mulatt(outputs,length2mask(sentence_len,word_size),att1.reshape(batch_size,context_size,-1))
        if self.config.bio_first:
            fused=torch.cat([mid_outputs1,mid_outputs2,tags1,tags2],-1)#b,c,c,2h
        else:fused=torch.cat([mid_outputs1,mid_outputs2,extra],-1)#b,c,c,2h
        fused=self.drop_layer(fused)
        logits_for_pairs=self.pair2(torch.tanh(self.pair1(fused))).squeeze(-1)#b,c,c
        logits_for_pairs=torch.sigmoid(logits_for_pairs)
#        context_mask=length2mask(context_len,context_size)
#        context_mask=context_mask.unsqueeze(1)*context_mask.unsqueeze(2)
#        loss2=self.binloss(logits_for_pairs,adjs)*context_mask
#        loss2=loss2.sum(-1).sum(-1).mean()
        return logits_for_class.cpu().data.numpy(),logits_for_pairs.cpu().data.numpy(),att1.reshape(batch_size,context_size,-1).cpu().data.numpy(),att2.cpu().data.numpy()
        