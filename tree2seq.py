import anytree
from treelib import Tree, Node
#from createdata import sememeDataset
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


def seq2tree(treeSeq):
    anyTreeMe = Tree()
    anyTreeMe.create_node(identifier=0,tag = "start", data="2083")
    temp = anyTreeMe.nodes[0]
    cnt = 1
    EdgeMap = {}

    for meId in range(1, len(treeSeq)):
        i = treeSeq[meId]
        if i != "end":
            anyTreeMe.create_node(identifier = cnt, tag = i, data = i,parent = temp)
            cnt += 1
            temp = anyTreeMe.nodes[cnt-1]

            
            if not temp.is_root():
                fatherId = anyTreeMe.parent(temp.identifier).data
                if EdgeMap.get(fatherId):
                    EdgeMap[fatherId].append(temp.data)
                else:
                    EdgeMap[fatherId] = [temp.data]
            
        else:
            if not temp.is_root():
                temp = anyTreeMe.parent(temp.identifier)  

    return anyTreeMe


def tree2seq(tree):
    # {"x":["", {"y":[...]}]}
    result = []
    def recuisiveChange(tree):

        if type(tree) == type(""):
            result.append(tree)
            result.append("end")
            #count += 2
        elif type(tree) == type({}):
            for key in tree.keys():
                result.append(key)
                #count += 1
                for cont in tree[key]:
                    recuisiveChange(cont)
                result.append("end")
    result.append("start")
    recuisiveChange(tree)
    return result


def dealChildTree(tree1,tree2,id1,id2):
    cnt = 1
    deal1 = tree1.nodes[id1]
    deal2 = tree2.nodes[id2]
    dealList1 = tree1.children(id1)
    dealList2 = tree2.children(id2)
    dealId2 = [i.data for i in dealList2]
    for i in range(len(dealList1)):
        for j in range(len(dealList2)):
                if dealList1[i].data == dealList2[j].data:
                    cnt += dealChildTree(tree1,tree2,dealList1[i].identifier, dealList2[j].identifier) 
                    break
    return cnt



def computeStrict(treeMe, treeAns,id2sememe = None, endtoken = "2084"):
    # list of str of ids

    anyTreeMe = Tree()
    anyTreeMe.create_node(identifier=0,tag = "me", data="2083")
    temp = anyTreeMe.nodes[0]
    cnt = 1
    Strict, Vertex,Edge = 0, 0, 0
    EdgeMap = {}

    for meId in range(1, len(treeMe)):
        i = treeMe[meId]
        if i != endtoken:
            if id2sememe != None:
                anyTreeMe.create_node(identifier = cnt, tag = id2sememe[int(i)], data = i,parent = temp)
            else:
                anyTreeMe.create_node(identifier = cnt, tag = i, data = i,parent = temp)
            cnt += 1
            temp = anyTreeMe.nodes[cnt-1]

            
            #compute vertex
            if i in treeAns:
                Vertex += 1
                # create edge map
            if not temp.is_root():
                fatherId = anyTreeMe.parent(temp.identifier).data
                if EdgeMap.get(fatherId):
                    EdgeMap[fatherId].append(temp.data)
                else:
                    EdgeMap[fatherId] = [temp.data]
            
        else:
            if not temp.is_root():
                temp = anyTreeMe.parent(temp.identifier)

    anyTreeAns = Tree()
    anyTreeAns.create_node(identifier=0,tag = "answer", data="2083")
    temp = anyTreeMe.nodes[0]
    cnt = 1
    
    for ansId in range(1, len(treeAns)):
        i = treeAns[ansId]
        if i != endtoken:
            if id2sememe != None:
                anyTreeAns.create_node(identifier = cnt, tag = id2sememe[int(i)], data = i,parent = temp)
            else:
                anyTreeAns.create_node(identifier = cnt, tag = i, data = i,parent = temp)
            cnt += 1
            temp = anyTreeAns.nodes[cnt-1]
            
            # compute edge
            if not temp.is_root():
                fatherId = anyTreeAns.parent(temp.identifier).data
                if EdgeMap.get(fatherId) and temp.data in EdgeMap[fatherId]:
                    Edge += 1
        else:
            if not temp.is_root():
                temp = anyTreeAns.parent(temp.identifier)
    #print(anyTreeMe)
    #print(EdgeMap)
    #exit()
    #print(anyTreeAns)
    len1 = anyTreeMe.size()
    len2 = anyTreeAns.size()
    assert(len2 > 1)
    cnt = dealChildTree(anyTreeMe,anyTreeAns,0,0)
    #print(cnt)
    P,R = 0,0
    if cnt > 1 and len1 > 1 and len2 > 1:
        P = (cnt -1) / (len1 -1)
        R = (cnt - 1) / (len2 - 1)
        Strict = 2 * P * R / (P + R)
    else:
        Strict = 0.0
        
    if Vertex > 0:
        VertexP,VertexR = Vertex / (len1 - 1), Vertex / (len2 - 1)
        Vertex = 2 * VertexP * VertexR / (VertexP + VertexR)
    else:
        Vertex = 0.0

    if Edge > 0:
        EdgeP,EdgeR = Edge / (len1 - 1), Edge / (len2 - 1)
        Edge = 2 * EdgeP * EdgeR / (EdgeP + EdgeR)
    else:
        Edge = 0.0
            
    return [Strict, Edge, Vertex, P, R], anyTreeMe, anyTreeAns


def computeDistance(stackList,id_other,id):
    dup = 0
    while dup < len(stackList[id]) and dup < len(stackList[id_other]) and stackList[id][dup] == stackList[id_other][dup]:
        dup += 1
    return len(stackList[id]) + len(stackList[id_other]) - 2 * dup


def computeBiasAndDepth(dictsSeqId, mask = 0):
    length = len(dictsSeqId[0]) - mask # len of a tree seq
    bias = torch.zeros(len(dictsSeqId),length,length, dtype=torch.long)
    depth = torch.zeros(len(dictsSeqId),length, dtype=torch.long)
    for treeSeqindex in range(len(dictsSeqId)):
        treeSeq = dictsSeqId[treeSeqindex]
        # for every seq in batch, init stack list and depthNow first
        stackList = [[] for _ in range(length)]
        depthNow = 0
        for id in range(length):
            # for each cont in seq
            if treeSeq[id] == 'start':
                # st
                stackList[id].append("st")
            elif treeSeq[id] == 'end':
                # end
                if stackList[id-1][-1][:3] == 'ba_':
                    stackList[id] = stackList[id-1][:-2]
                    depthNow -= 2
                else:
                    stackList[id] = stackList[id-1][:-1]
                    depthNow -= 1
            else:
                if stackList[id-1][-1][:3] == 'ba_':
                    stackList[id] = stackList[id-1][:-1] + ['ba_'+str(treeSeq[id]),str(treeSeq[id])]
                    depthNow += 1
                else:
                    stackList[id] = stackList[id-1] + ['ba_'+str(treeSeq[id]),str(treeSeq[id])]
                    depthNow += 2
            #compute depth    
            depth[treeSeqindex][id] = depthNow
                
            # compute distance
            bias[treeSeqindex,id,id] = 0
            for id_other in range(id):
                dis = computeDistance(stackList,id_other,id)
                bias[treeSeqindex,id,id_other] = dis
                bias[treeSeqindex,id_other,id] = dis
                
        # try [['2083','1','2','2084','3','4','2084','5','2084','2084','2084']] for example
            #print(stackList)
        #print(bias)
            #print(depth)
    return bias, depth


if __name__ == "__main__":
    '''dealDataset = torch.load("./datas/data.pkl")
    c = 0
    for cont in dealDataset:
        print(cont[2][0])
        result = tree2seq(cont[2][0])
        print(result)
        print("\n")
        c += 1
        if c > 100:
            break'''

    computeStrict(['2083', '50', '888', '2084', '342', '2084', '343', '9', '2084', '2084', '344', '9', '2084', '2084', '2084'],['2083', '50', '888', '2084', '342', '2084', '341', '9', '2084', '2084', '345', '9', '2084', '2084', '2084'])