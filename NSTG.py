import demjson
import re
import math
import copy

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
import numpy as np
import warnings
#import preprocessor
import json
import tqdm
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

from createdata import sememeDataset
from transformers import BertModel, BertTokenizer

import anytree
import treelib
from treelib import Tree, Node
import tree2seq

torch.cuda._initialized = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sentence_transformers import SentenceTransformer

"""
NSTG is intuitional.
In this file, we implement the NSTG method, including the model and the test. You can only use this file to predict the sememe tree. 
This method is a lot different with trnasformer based method.
"""

TEST_SET_NAME = ""
TRAIN_SET_NAME = ""
BERT_PATH = ""
SEMEME_NAME_PATH= ""
NSTG_RESULT_PATH = ""


def sigmoid(x):
    return 1 / (1 + math.e ** -x)
            
class candidate_tree():
    """
    ### This is the base class of sememe tree used in NSTG predicition
    - `self.size` means the size of the tree
    - `self.score` means the predicition score of exisiting result
    - `self.tree` is a `treelib` object.
    - `self.all_nodes` is the set of terminal nodes (dynamic changed), used to speed up the NSTG prediction
    """    
    def __init__(self, sememe_id, size = 1, score = 1.0, sentence = None, other_tree = None):
        if (other_tree != None):
            self.tree = other_tree.tree.subtree(other_tree.root.identifier)
            self.root = self.tree.nodes[0]
            self.sentence = other_tree.sentence
            self.size = other_tree.size
            self.score = other_tree.score

            self.all_nodes = []
            for node in other_tree.all_nodes:
                self.all_nodes.append(self.tree.nodes[node.identifier])
        else:
            self.sentence = sentence
            self.size = size
            self.score = score
            self.tree = Tree()
            self.tree.create_node(identifier = 0, data=1.0, tag = sememe_id["start"])
            self.root = self.tree.nodes[0]
            self.all_nodes = [self.root]
            
    def add_node(self, terminal_node, child_id,score):
        terminal_node_id = terminal_node.identifier
        terminal_node_ = self.tree.nodes[terminal_node_id]
        #print(anytree.RenderTree(self.tree.nodes[0], style=anytree.AsciiStyle()).by_attr())
        #node = anytree.find(self.root, lambda node: node.data == terminal_id)
        self.tree.create_node(identifier = self.size,data = terminal_node_.data * score.data.item(), parent = terminal_node_, tag = child_id)
        child_node = self.tree.nodes[self.size]
        self.all_nodes.append(child_node)
        if len(self.tree.children(terminal_node_id)) > 0:
            #a new tree path
            #self.score *= sigmoid(child_node.data)
            self.score *= child_node.data
        else:
            #self.score = self.score / sigmoid(terminal_node_.data) * sigmoid(child_node.data)
            self.score = self.score  * score.data.item()
        self.size += 1
    


class NSTG(nn.Module):
    def __init__(self, trainSetPath, d = 0.9):
        super(NSTG, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizer, BERT_PATH)    
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.dropout = nn.Dropout(p = 0.1)
        self.device = torch.device("cuda")
        self.bert = model_class.from_pretrained(pretrained_weights).to(self.device)
        
        self.trainData = torch.load(trainSetPath)
        
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        print(len(self.trainData))
        
        
        self.d = d
        self.eee = [math.pow(d,i) for i in range(10000)] + [0.0 for i in range(100000)]
        
        self.sememe2id = {}
        self.id2sememe = []

        names = torch.load(SEMEME_NAME_PATH)[1:]
        self.id2sememe = names
        self.id2sememe.append("start")
        self.id2sememe.append("end")
        for id in range(len(self.id2sememe)):
            self.sememe2id[self.id2sememe[id]] = id
        self.sememe2id["end"] = len(self.id2sememe) - 1
        self.sememe2id["start"] = len(self.id2sememe) - 2
        assert(self.sememe2id["start"] == 2083)
        assert(self.sememe2id["end"] == 2084)
        
        self.neighbor_metric = self.create_candidate()
        
        self.CLS_metric = self.make_CLS_metric()
        print(self.CLS_metric.size())
        
        # give a number for each sememe
        
    
    def make_CLS_metric(self):
        with torch.no_grad():
            #! hard coded
            result = torch.zeros([0,768])
            for count in tqdm.tqdm(range(len(self.trainData) // 32 + 1)):
                batch_sentences = []
                for i in range(32 * count , min(len(self.trainData), 32*(count+1))):
                    batch_sentences.append(self.trainData[i][1][0])
                #print(batch_sentences)
                sentence_embeddings = self.model.encode(batch_sentences)
                sentence_embeddings = torch.from_numpy(sentence_embeddings)
                #print(sentence_embeddings.shape)
                #exit()
                result = torch.cat((result, sentence_embeddings), 0)
        return result
    
    def compute_CLS_similar(self, sentence):
        """we use pretrained sentence transformer to compute the `cosine similarity` of input definition and existing definitions.

        Args:
            sentence (string): input synset definition

        Returns:
            [train_set_len,torch.float]: cosine similarity
        """        
        with torch.no_grad():
            batch_sentences = [sentence]
            sentence_embeddings = self.model.encode(batch_sentences)
            sentence_embeddings = torch.from_numpy(sentence_embeddings)
            return torch.cosine_similarity(sentence_embeddings, self.CLS_metric, dim=1)
    
    '''def make_CLS_metric(self):
        with torch.no_grad():
            #! hard coded
            result = torch.zeros([0,768]).to(self.device)
            for count in tqdm.tqdm(range(len(self.trainData) // 32 + 1)):
                batch_sentences = []
                for i in range(32 * count , min(len(self.trainData), 32*(count+1))):
                    batch_sentences.append(self.trainData[i][0])
                batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                    pad_to_max_length=True) 
                input_ids = torch.tensor(batch_tokenized['input_ids']).to(self.device)
                attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(self.device)
                bert_out = self.bert(input_ids, attention_mask=attention_mask)[0][:,0,:]
                input_ids.to("cpu")
                attention_mask.to("cpu")
                #print(bert_out)
                #print(bert_out.size())
                #bert_output = bert_output.to("cpu")
                result = torch.cat((result, bert_out), 0)
        return result
    
    def compute_CLS_similar(self, sentence):
        with torch.no_grad():
            batch_sentences = [sentence]
            batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                pad_to_max_length=True) 
            input_ids = torch.tensor(batch_tokenized['input_ids']).to(self.device)
            attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(self.device)
            cls_me = self.bert(input_ids, attention_mask=attention_mask)[0][:,0,:]
            input_ids.to("cpu")
            attention_mask.to("cpu")
            return torch.cosine_similarity(cls_me, self.CLS_metric, dim=1)'''
            
    def write_child(self, tree, node):
        result = []
        result.append(node.tag)
        for child in tree.children(node.identifier):
            result = result + self.write_child(tree, child)
        result.append(self.sememe2id["end"])
        return result
    
    def to_tree_str(self, cont):
        return self.write_child(cont.tree, cont.root)[:-1]
    
    def predict(self, input_sentence, max_len = 20, beam_size = 50, try_count = 5, restricted = None):
        if restricted != None:
            self.restrict = torch.zeros(len(self.sememe2id))
            for id in restricted:
                self.restrict[int(id)] = 1
        else:
            self.restrict = torch.ones(len(self.sememe2id))
            
        beam_list = []
        self.cosine_sim = self.compute_CLS_similar(input_sentence)
        #print(self.cosine_sim)
        _, rank = self.cosine_sim.sort(descending=True)

        self.cosine_sim_rank = [0 for id in range(self.cosine_sim.shape[0])]
        for id in range(self.cosine_sim.shape[0]):
            self.cosine_sim_rank[rank[id]] = self.eee[id]
        beam_list.append(candidate_tree(self.sememe2id, sentence = input_sentence))
        temp_list = []
        
        now_length = 1
        end = 0
        while (now_length < max_len and (end == 0)):
            #print(f"******************{now_length}")
            # for each candidate, convert to 5
            temp_list = []
            for tree in beam_list:
                temp_list = temp_list + self.gen_more_trees(tree,  beam_size)
            
            temp_list.sort(key = lambda tr:tr.score, reverse = True)
            number = min(len(temp_list), beam_size)
            beam_list = temp_list[:number]
            now_length = now_length + 1
            
            end = 1
            score1 = beam_list[0].score
            for cont in beam_list:
                if abs(score1 - cont.score) > 1e-10:
                    end = 0
                    break
            
            '''for i in range(5):
                beam_list[i].tree.show()
                print(f"score: {beam_list[i].score}")
                print(f"size: {beam_list[i].tree.size()}")'''
        return beam_list[0], self.to_tree_str(beam_list[0])
    
    def gen_more_trees(self, root_tree, beam_size = 50):
        # beam search more trees with root tree
        #print(root_tree.score)
        if root_tree.size > 1:
            candidate_trees = [root_tree]
        else:
            candidate_trees = []
        #print(cosine_sim.size())
        for terminal_node in root_tree.all_nodes:
            terminal_id = int(terminal_node.tag)
            skip = 0
            for child in self.neighbor_metric[terminal_id]:
                if self.restrict[child] == 0:
                    continue
                for nodes in root_tree.all_nodes:
                    if int(nodes.tag) == child:
                        skip = 1
                        break
                if skip:
                    continue
                # gen a children-tree with terminal->child
                score = 0.0
                for other_id in self.neighbor_metric[terminal_id][child]:
                    # decending rank r
                    score += (self.cosine_sim[other_id] * self.cosine_sim_rank[other_id])
                #if (score > 1):
                #    print(score)
                new_tree = candidate_tree(self.sememe2id, other_tree=root_tree)
                new_tree.add_node(terminal_node,child, score)
                candidate_trees.append(new_tree)
        #candidate_trees.sort(key = lambda tree: tree.score, reverse= True)
        #candidate_trees = candidate_trees[:min(len(candidate_trees), beam_size)]
        '''for i in candidate_trees:
            i.tree.show()
            print(f"score: {i.score}")
        exit()'''
        #print(root_tree.score)

        return candidate_trees
        
    def append_candidate(self, result, father_id, child_id, candidate_id):
        if result[father_id].get(child_id):
            if not (candidate_id in result[father_id][child_id]):
                result[father_id][child_id].append(candidate_id)
        else:
            result[father_id][child_id] = [candidate_id]
        return result
            
    def create_candidate(self):
        # result[father_id]{"child_id":[candidate_id]}
        
        result = [{} for i in range(len(self.id2sememe))]
        
        for contid in tqdm.tqdm(range(len(self.trainData))):
            cont = self.trainData[contid]
            #print(cont)
            father_stack = [self.sememe2id[cont[2][0][0]]]
            father_sememe_id = self.sememe2id[cont[2][0][0]]
            for id in range(1, len(cont[2][0])):
                child_sememe_id = self.sememe2id[cont[2][0][id]]
                if (child_sememe_id == self.sememe2id['end']):
                    #pop_stack
                    father_stack.pop()
                    father_sememe_id = father_stack[-1]
                else:
                    #father_id -> child_id
                    result = self.append_candidate(result, father_sememe_id, child_sememe_id, contid)
                    father_sememe_id = child_sememe_id
                    father_stack.append(father_sememe_id)
            assert(father_sememe_id == self.sememe2id[cont[2][0][0]])
        return result

            
            
    
    
    
def test():
    testData = torch.load(TEST_SET_NAME)
    cc =  len(testData)
    temp = []
    shunxu = list(range(len(testData)))
    random.seed(2077)
    #torch.manual_seed(2077)
    random.shuffle(shunxu)
    model = NSTG(TRAIN_SET_NAME, d = 0.9)
    model.to(torch.device("cuda"))
    
    acc1,acc2,acc3,acc4 = 0.0,0.0,0.0,0.0
    total1,total2 = 0.0, 0.0
    bar = tqdm.tqdm(range(cc))
    for id in bar:
        #print("**************")
        try:
            cont = testData[shunxu[id]]
            ans = [str(model.sememe2id[i]) for i in cont[2][0]] + ["2084"]
            a,a_str = model.predict(cont[1][0], max_len=20, beam_size= 10, try_count=5, restricted = ans)
            #a,a_str = model.predict(cont[1][0], max_len=20, beam_size= 10, try_count=5)
            out = [str(i) for i in a_str]
            cont.append(out)
            scores,meT,ansT = tree2seq.computeStrict(out,ans,id2sememe = model.id2sememe)
            
            #ans.remove("2083")
            #ans.remove("2084")
            #out.remove("2083")
            #out.remove("2084")
            for i in ans.copy():
                if i == "2083" or i == "2084":
                    ans.remove(i)
            for i in out.copy():
                if i == "2083" or i == "2084":
                    out.remove(i)
            #print(ans)
            #print(out)
            score1 = sentence_bleu([ans], out,smoothing_function=SmoothingFunction().method1)
            
            acc1 += score1
            total1 += 1
            acc2 += scores[0]
            acc4 += scores[2]
            acc3 += scores[1]
            total2 += 1
            #print(cont[1][0])
            #print(f"this score :{format(score1, '.4f')} {format(scores[0], '.4f')} {format(scores[1], '.4f')} {format(scores[2], '.4f')}", )
            #print(meT)
            #print(ansT)
            cont.append([score1] + scores)
            temp.append(cont)
            if total1 != 0 and total2 != 0:
                bar.set_description(f"{format(acc1 / total1, '.4f')}, {format(acc2/total2, '.4f')}, {format(acc3/total2, '.4f')}, {format(acc4/total2, '.4f')}")
            elif total2 != 0:
                bar.set_description(f"{0.0}, {format(acc2/total2, '.4f')}, {format(acc3/total2, '.4f')}, {format(acc4/total2, '.4f')}")
        except:
            total1 += 1
            total2 += 1
    
    torch.save(temp,NSTG_RESULT_PATH)

if __name__ == "__main__":
    #a = torch.load("./data/sstg_out.pkl")
    #print(a)
    test()