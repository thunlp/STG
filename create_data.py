import demjson
import re
#导入相关模块
from torch.utils.data import DataLoader,Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import json
import tree2seq 

def dealsememe(match):
    kuohaoC = 0
    ifRec = 1
    #print(match.group(0))
    
    for i in range(len(match.group(0))):
        if match.group(0)[i] == "{":
            kuohaoC += 1
        elif match.group(0)[i] == "}":
            kuohaoC -= 1
        if kuohaoC == 0:
            if i != len(match.group(0)) -1:
                ifRec = 0
            break
        
    if ifRec:
        ifmaohao = 0
        for i in range(1,len(match.group(0))):
            kuohaoC = 0
            if match.group(0)[i] == "{":
                break
            if match.group(0)[i] == ":":
                ifmaohao = i
                break
        if ifmaohao != 0:
            a = "{\""+ match.group(0)[1:i] + "\":[" + re.sub("{.*}",dealsememe,match.group(0)[i+1:-1]) + "]}"
            #print(a)
            return a
        else:
            #print("\"" + match.group(0)[1:-1] + "\"")
            return "\"" + match.group(0)[1:-1] + "\""
    else:
        #print(match.group(0))
        count = 0
        for i in range(len(match.group(0))):
            if match.group(0)[i] == "{":
                count += 1
            elif match.group(0)[i] == "}":
                count -= 1
            if match.group(0)[i] == "," and count == 0:
                return re.sub("{.*}",dealsememe,match.group(0)[0:i]) + "," + re.sub("{.*}",dealsememe,match.group(0)[i+1:])
        


class sememeDataset(Dataset):
    def __init__(self,start,end,sorts):
        self.data = []
        treeDict = {}
        defDict = {}
        trees = open("./data/synsetStructed.txt",'r',encoding="utf-8")
        defs = open("./data/synsetDef.txt",'r',encoding="utf-8")
        
        good,bad = 0,0
        name = ""
        for line in defs.readlines():
            line = line.strip()
            if line == "":
                continue
            if line.startswith("bn:"):
                name = line
                continue
            ls = line.split("	")
            if int(ls[0]) == 0 or (defDict.get(name,0) != 0):
                continue
            
            for defineId in range(1,len(ls)):
                define = ls[defineId]
                if (defDict.get(name,0) != 0):
                    if define not in defDict[name]:
                        defDict[name].append(define)
                else:
                    defDict[name] = [define]
            #print(defDict[name])
            good += 1
        
        print(f"definations parsing: good:{good}")
        good,bad = 0,0
        for content in trees.readlines()[start:end]:
            text = content.strip().split("	")[2][1:]
            name = content.strip().split("	")[1]
            b = text.split(";")
            try:
                for a in b:
                    a = re.sub("[^:,]*?=","",a)
                    a = a.replace("}{","},{")
                    #print(a)
                    a = re.sub("{.*}",dealsememe,a)
                    #print(a)
                    #a = demjson.decode(a)
                    a = json.loads(a)
                    #print(type(a))
                    good += 1
                    if (treeDict.get(name,0) != 0) :
                        treeDict[name].append(a)
                    else:
                        treeDict[name] = [a]
            except:
                bad += 1
                continue
        print(f"sememeTree parsing: good:{good}, bad:{bad}")
        
        print(f"defLen: {len(defDict)}, treeLen: {len(treeDict)}" )
        
        for key in treeDict:
            if defDict.get(key,0) == 0:
                continue
            a = [tree2seq.tree2seq(i) for i in treeDict[key]]

            biasS, depthS = [],[]
            for i in a:
                bias, depth = tree2seq.computeBiasAndDepth([i], mask = 1) 
                biasS.append(bias)
                depthS.append(depth)
            self.data.append([key,defDict[key], a, biasS, depthS])
            
        if (sorts == 1):
            self.data.sort(key = lambda x : len(x[2][0]))
        print(f"dataLen: {len(self.data)}")
            
        
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    def sorts(self):
        self.data.sort(key = lambda x : len(x[2][0]))
    
def sorts(sememeSet):
    sememeSet.sort(key = lambda x : len(x[2][0]))
    
if __name__ == "__main__":
    # dealDataset = sememeDataset(0,-1,1)
    # torch.save(dealDataset,"./data/allData.pkl")
    
    trainSet = sememeDataset(0,35000,1)
    torch.save(trainSet,"./data/trainSet.pkl")
    validSet = sememeDataset(35000,36000,0)
    torch.save(validSet,"./data/validSet.pkl")
    testSet = sememeDataset(36000,-1,0)
    torch.save(testSet,"./data/testSet.pkl")
    
    # load and try the dataset
    # dealDataset = torch.load("./data/allData.pkl")
    # for cont in dealDataset:
    #     if cont[0] == "bn:00009609n":
    #         print(cont)
    #         break
    # trainSet,testSet = torch.utils.data.random_split(dealDataset,[int(0.8*len(dealDataset)),len(dealDataset) - int(0.8*len(dealDataset))])
    

    # see the detail in preprocessed data
    # print(dealDataset[2])
    # print(dealDataset[100])
    # print(dealDataset[30000])
    # print(dealDataset[40000])