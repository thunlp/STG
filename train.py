import demjson
import re

from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
from skimage import io,transform
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import transformers as tfs
import math
import random
import os
import sys
import numpy as np
import warnings
#import preprocessor
import tqdm
import create_data
import tree2seq 
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

from create_data import sememeDataset
from transformers import BertModel, BertTokenizer
from model import TreePred
import argparse


torch.cuda._initialized = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_SET_PATH = ""
VALID_SET_PATH = ""
TEST_SET_PATH = ""
MODEL_SAVE_PATH = ""
MAX_EPOCH = 200
LEARNING_RATE = 1e-5
PRETRAINED = 1
MASK=1
TUPE = 0
SEQ = False
DEPTH_METHOD = ""
BIAS_METHOD = ""
MODEL_NAME = ""

def parse_args():

    description = \
    '''
        args is defined bellow:
    '''         
    parser = argparse.ArgumentParser(description=description)       
    
    parser.add_argument('--learning_rate',help = "learning rate", required=False,type=float,default=1e-5)    
    
    parser.add_argument('--max_epoch',help = "max training epochs", required=False,type=int,default=200)   
    parser.add_argument('--pretrained',help = "use pretrained sememe embeddings", required=False,type=int,default=1)       
    
    parser.add_argument('--mask',help = "use candidate mask", required=False,type=int,default=1) 
    
    parser.add_argument('--tree_attention',help = "use tree attention method", required=False,type=int,default=1)    
    
    parser.add_argument('--depth_method',help = "", required=False,type=str,default="depth")   

    parser.add_argument('--bias_method',help = "", required=False,type=str,default="distance")   
    
    parser.add_argument('--sequence',help = "use sequence encoding result", required=False,type=bool,default=True)    
    
    parser.add_argument('--train_set_path',help = "", required=False,type=str,default="./data/trainSet.pkl")      
    parser.add_argument('--test_set_path',help = "", required=False,type=str,default="./data/testSet.pkl")      
    parser.add_argument('--valid_set_path',help = "", required=False,type=str,default="./data/validSet.pkl")     
    parser.add_argument('--model_save_path',help = "", required=False,type=str,default="./model")       
    
    parser.add_argument('--model_name',help = "how to name your model", required=False,type=str)                    
    args = parser.parse_args()                      
    return args


def active_bytes():
    stats = torch.cuda.memory_stats()
    current_active_byte =  stats['active_bytes.all.current']
    return current_active_byte


def testFirst(model, testSet = "./", beamWidth = 1):
    testData = torch.load(testSet)
    shunxu = list(range(len(testData)))
    random.seed(2077)
    #torch.manual_seed(2077)
    random.shuffle(shunxu)
    model = model.eval()
    acc1,acc2,acc3,acc4 = 0.0,0.0,0.0,0.0
    total1,total2 = 0.0, 0.0
    with torch.no_grad():
        bar = tqdm.tqdm(range(1000))
        
        for id in bar:
            torch.cuda.empty_cache() 
            cont = testData[shunxu[id]]
            #print(cont)

            ans = [2083]+[model.sememe2id[i] for i in cont[2][0]][1:] + [2084]
            ans = [str(i) for i in ans]
            out = model.predict([cont[1][0]],maxLen=15,beamWidth = beamWidth)
            #ls = out.argmax(dim=1).to("cpu").numpy().tolist()
            ls = [str(i) for i in out]

            #print(ls)
            #print([ans])
            scores,_,_ = tree2seq.computeStrict(ls,ans)
            
            ans.remove("2083")
            ans.remove("2084")
            ls.remove("2083")
            ls.remove("2084")
            score1 = sentence_bleu([ans], ls, smoothing_function=SmoothingFunction().method1)
            
            #print(score)
            acc1 += score1
            acc2 += scores[0]
            acc4 += scores[2]
            total1 += 1
            acc3 += scores[1]
            total2 += 1
                
            if total1 != 0 and total2 != 0:
                bar.set_description(f"{format(acc1 / total1, '.4f')}, {format(acc2/total2, '.4f')}, {format(acc3/total1, '.4f')}, {format(acc4/total1, '.4f')}")

        #print(f"score: {acc / total}")
        return [acc1 / total1, acc2 / total2, acc3/total2, acc4 / total2]
    

    
def myCollate(batch):
    bns = []
    texts,dicts, bias, depth= [],[], [], []
    for cont in batch:
        bns.append(cont[0])
        texts.append(cont[1][0])
        dicts.append(cont[2][0])
        #bias.append(cont[3][0])
        #depth.append(cont[4][0])
        #assert(len(cont[4][0]) == len(depth[0]))
        #assert(len(cont[3][0]) == len(bias[0]))
    #try:
        #bias = torch.tensor([t.numpy() for t in bias],dtype=torch.long).squeeze(1)
        #depth = torch.tensor([t.numpy() for t in depth] ,dtype=torch.long).squeeze(1)
    #except:
        # first len != last len: jump in the train
    #    pass
    return texts,dicts,bias,depth

            

def train(epoch = MAX_EPOCH, batchSize = 32, trainSet = TRAIN_SET_PATH, testSet = TRAIN_SET_PATH, validSet = VALID_SET_PATH, modelName=MODEL_NAME,lr = LEARNING_RATE):
    
    
    trainData = torch.load(trainSet)
    
    
    trainLoader=DataLoader(dataset=trainData,batch_size=batchSize,shuffle=False,num_workers=0,collate_fn = myCollate)
        
    model = TreePred(wordDim = 768, maxPosEmbed = 50,sememeDim = 768, head=8,attentionLayer = 8,hiddenDim = 128, pretrained = PRETRAINED, TUPE=TUPE, MASK=MASK, seq = SEQ, depthMethod=DEPTH_METHOD, biasMethod=BIAS_METHOD)
    model.to(device)
    
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epochId in range(0,epoch):
        model.to(device)
        model.device = device
        lens = len(trainLoader)
        bar = tqdm.tqdm(enumerate(trainLoader), total=lens)
        model = model.train()
        for id, (texts,dicts,bias,depth) in bar:
            if len(dicts[0]) != len(dicts[-1]):
                # make sure treeLen equals in a batch
                continue
            out,dictsSeqId = model(texts,dicts)
            # out : [batchSize, treeLen - 1, sememeCount]

            out = out.view(-1,out.shape[2])

            dictsSeqId = torch.tensor(dictsSeqId,dtype=torch.long)[:,1:].to(device)
            dictsSeqId = dictsSeqId.view(-1)


            loss = criterion(out,dictsSeqId)
            #loss = criterion(out,target)
            
            bar.set_description(f"[#{epochId+1}]loss: {format(float(loss),'.6f')},cuda: {active_bytes()}")
        
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            out.to("cpu")
            dictsSeqId.to("cpu")
            torch.cuda.empty_cache() 
        if (epochId % 10 == 0):
            # test model
            #model.to('cpu')
            #model.device = torch.device('cpu')
            results = testFirst(model,testSet)
            f = open("info_"+modelName+"_test.txt",'a',encoding='utf-8')
            f.write(f"{epochId},   {format(results[0],'.5f')},  {format(results[1],'.5f')}, {format(results[2],'.5f')}, {format(results[3],'.5f')}\n")
            f.close()
            
            results = testFirst(model,validSet)
            f = open("./train_info/info_"+modelName+"_validation.txt",'a',encoding='utf-8')
            f.write(f"{epochId},   {format(results[0],'.5f')},  {format(results[1],'.5f')}, {format(results[2],'.5f')}, {format(results[3],'.5f')}\n")
            f.close()
            
            #save model
            model.to("cpu")
            model.eval()
            if not os.path.exists(MODEL_SAVE_PATH+f"/{modelName}_{format(epochId,'03d')}/"):
                os.mkdir(MODEL_SAVE_PATH+f"/{modelName}_{format(epochId,'03d')}/")
            model.saveBert(MODEL_SAVE_PATH+f"/{modelName}_{format(epochId,'03d')}/")
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict()['param_groups'], 'epoch':epochId}
            torch.save(state, MODEL_SAVE_PATH+f"/{modelName}_{format(epochId,'03d')}/modelWeight.pt")
            

    
    
if __name__ == "__main__":
    args = parse_args()
    TRAIN_SET_PATH = args.train_set_path
    VALID_SET_PATH = args.valid_set_path
    TEST_SET_PATH = args.test_set_path
    MODEL_SAVE_PATH = args.model_save_path
    MAX_EPOCH = args.max_epoch
    LEARNING_RATE = args.learning_rate
    PRETRAINED = args.pretrained
    MASK = args.mask
    TUPE = args.tree_attention
    SEQ = args.sequence
    DEPTH_METHOD = args.depth_method
    BIAS_METHOD = args.bias_method
    MODEL_NAME = args.model_name
    #print(f"begin training {sys.argv[1]}")
    train()
    #demo()