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
from create_data import sememeDataset



def makeCandidate(trainName, testName,sememeName,candiName,sememeSaveName):
    #test weather candidates in test have shown up in train 
    train, test = torch.load(trainName),torch.load(testName)
    candi = {}
    sememes = [0,]
    good,bad = 0,0
    #print(train[0])
    #print(train[10000])
    #print(train[20000])
    def recursiveDeal(tree,sememe):
        #tree is a dict or a str
        if type(tree) == type(""):
            sememes[0] += 1
            if tree not in sememes:
                sememes.append(tree)
            if candi.get(sememe,0) != 0:                    
                use = 0
                for id in range(len(candi[sememe])):
                    if tree == candi[sememe][id][0]:
                        candi[sememe][id][1] += 1
                        use = 1
                        break
                if use == 0:
                    candi[sememe].append([tree,1])        
            else:
                candi[sememe] = [[tree,1]]
            return 
                            
        for key in tree.keys():
            sememes[0] += 1
            if key not in sememes:
                sememes.append(key)
            if (key == sememe):
                print(tree)
                
            if candi.get(sememe,0) != 0:
                use = 0
                for id in range(len(candi[sememe])):
                    if key == candi[sememe][id][0]:
                        candi[sememe][id][1] += 1
                        use = 1
                        break
                if use == 0:
                    candi[sememe].append([key,1])         
            else:
                candi[sememe] = [[key,1]]
                
            if type(tree[key]) == type([]):
                for child in tree[key]:
                    recursiveDeal(child,key)
            else:
                print(tree[key])
                
    def recursiveCheck(tree,sememe,good,bad):
        #tree is a dict or a str
        if type(tree) == type(""):
            if tree not in sememes:
                sememes.append(tree)

            if candi.get(sememe,0) != 0:                    
                use = 0
                for id in range(len(candi[sememe])):
                    if tree == candi[sememe][id][0]:
                        use = 1
                        good += 1
                        break
                if use == 0:
                    bad += 1
            else:
                bad += 1
            return good,bad
                            
        for key in tree.keys():
            if key not in sememes:
                sememes.append(key)
            if candi.get(sememe,0) != 0:
                use = 0
                for id in range(len(candi[sememe])):
                    if key == candi[sememe][id][0]:
                        use = 1
                        good += 1
                        break
                if use == 0:
                    bad += 1
            else:
                bad += 1
                
            if type(tree[key]) == type([]):
                for child in tree[key]:
                    good,bad = recursiveCheck(child,key,good,bad)
            else:
                print(tree[key])
                
        return good,bad
                    
                                      
    for cont in train:
        name,defs,trees = cont[0]," ".join(cont[1]),cont[2]
        for subtree in trees:
            recursiveDeal(subtree,"start")
        #break
        
    def score(x):
        return x[1]
    for key in candi.keys():
        candi[key].sort(key = score,reverse = True)
    #print(candi["research|研究"])
    #print(candi["write|写"])
    #print(candi["compile|编辑"])
    totalT = 0
    badT = 0
    for cont in test:
        name,defs,trees = cont[0]," ".join(cont[1]),cont[2]
        for subtree in trees:
            goodf,badf = good,bad
            good,bad = recursiveCheck(subtree,"start",good,bad)
            totalT += 1
            if bad != badf:
                badT += 1
    print(f"{len(totalT)}, {len(badT)}")
            
    torch.save(candi,candiName)
    torch.save(sememes,sememeSaveName)
            
            
    print(f"{len(sememes)} sememes total, appear {sememes[0]} times, {len(candi)} sememes have branches")
    print(f"known rules: {good}, unknown rules: {bad}")
    #break
        
        

if __name__ == "__main__":
    makeCandidate("./data/trainSet.pkl","./data/testSet.pkl","./data/SememeFile.txt","./data/candi.pkl","./data/sememes.pkl")