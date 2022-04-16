from hashlib import new
from json import decoder
import demjson
import re

from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
from torch.autograd import Variable
import transformers as tfs
import copy
import math
import random
import os
import numpy as np
import warnings
#import preprocessor
import os
import json
import tqdm
import create_data
import tree2seq 

from create_data import sememeDataset
from transformers import BertModel, BertTokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME

v_o = None

BERT_PATH = ""
SEMEME_PAIR_PATH = ""
SEMEME_NAME_PATH = ""
SEMEME_EMBEDDING_PATH = ""

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    

def clones(module, N):
    # this method allow we clone a layer with different variable
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def attention(query, key, value, mask=None, dropout=None, attn_pos = None, bias = None):
    "Compute 'Scaled Dot Product Attention'"
    # query,key,value: tensor [batchSize, nHead, len, wordDim // nHead] present the tree 
    if attn_pos != None:
        # relative TUPE attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(2 * d_k)
        
        # add TUPE postion info here
        scores  = scores + attn_pos
        # scores: [batchSize, nHead, len ,len] 
        # mask need: [batchSize,hHead, len ,len]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        global v_o
        if v_o == None:
           v_o = scores
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        #print(p_attn.size())
        #print(attn_pos.size())
        return torch.matmul(p_attn, value), p_attn      
    
    else:
        # here is the normal attention score computation
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        
        # scores: [batchSize, nHead, len ,len] 
        # mask need: [batchSize,hHead, len ,len]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

def attentionRel(query, key, value, mask=None, dropout=None, bias = None):
    "Compute 'Scaled Dot Product Attention'"
    # query,key,value: tensor [batchSize, nHead, len, wordDim % nHead] present the tree 
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(2 * d_k)
    if bias != None:
        scores += bias
    # scores: [batchSize, nHead, len ,len] 
    # mask need: [batchSize,hHead, len ,len]
    # if mask is not None:
    #     scores = scores.masked_fill(mask == 0, -1e9)
    return scores

class MultiHeadedAttentionRel(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionRel, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.d_model = d_model
        self.h = h
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.pos_encoding = nn.Parameter(torch.Tensor(1,1000,d_model))
        nn.init.xavier_uniform_(self.pos_encoding)
        
    def makePostionMat(self, postionMat):
        """convert the position describe to position embedding for relative attention

        Args:
            postionMat (torch_tensor[batch_size,sent_len])

        Returns:
            torch_tensor[batch_size,pos_emb_dim]: the realtive position matrix used for TUPE attention
        """        ''''''
        mat = torch.zeros(postionMat.shape[0],postionMat.shape[1],self.d_model)
        for batch in range(postionMat.shape[0]):
            for pos in range(postionMat.shape[1]):
                mat[batch][pos] = self.pos_encoding[0,postionMat[batch,pos]]
        return mat
    
    def forward(self, query, key, value, mask=None, postionMat = None, bias = None,device = None):
        "Implements Figure 2"
        #query,key value: [batchSize, Len, d_model]
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            if postionMat == None:
                mask = mask[:1,:,:,:]
        nbatches = query.size(0)
        #compute postion embedding attention

        if postionMat == None:
            queryPos, keyPos, valuePos = \
                [l(x).view(1, -1, self.h, self.d_k).transpose(1, 2)
                for l, x in zip(self.linears, (self.pos_encoding[:,:query.size(1),:],self.pos_encoding[:,:key.size(1),:], self.pos_encoding[:,:value.size(1),:]))]
            attn_pos_alpha = attentionRel(queryPos, keyPos, valuePos, mask=mask, 
                                 dropout=self.dropout)
            attn_pos_alpha = attn_pos_alpha.repeat(nbatches,1,1,1)
        else:
            mat = self.makePostionMat(postionMat).to(device)
            queryPos, keyPos, valuePos = \
                            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (mat,mat, value))]               
            if bias != None:
                bias = bias.to(device)
                attn_pos_alpha = attentionRel(queryPos, keyPos, valuePos, mask=mask, 
                                    dropout=self.dropout, bias = bias)
                bias.to("cpu")
            else:
                attn_pos_alpha = attentionRel(queryPos, keyPos, valuePos, mask=mask, 
                                    dropout=self.dropout)
            mat.to("cpu")
            postionMat.to("cpu")

        return attn_pos_alpha

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None, attn_pos = None, bias = None):
        "Implements Figure 2 in attention is all you need"
        #query,key value: [batchSize, Len, wordDim]
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
            
            
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout, attn_pos = attn_pos,bias = bias)
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        x = self.linears[-1](x)
        return self.linears[-1](x)
    
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
 
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

    
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask, attn_pos = None,attn_pos_mem = None, bias = None):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask,attn_pos, bias = bias))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, attn_pos_mem))
        return self.sublayer[2](x, self.feed_forward)
    
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    # N=6
    def __init__(self, layer, N = 6):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask, attn_pos = None,attn_pos_mem = None, bias = None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, attn_pos,attn_pos_mem, bias = bias)
        return self.norm(x)
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

    

    

class TreePred(nn.Module):
    def __init__(self,wordDim = 768, maxPosEmbed = 50,sememeDim = 768,attentionLayer = 8, head = 8, hiddenDim = 128,pretrained = True, TUPE = False, MASK = True, seq = False, depthMethod = 'none', biasMethod = 'none'):
        """_summary_

        Args:
            wordDim (int, optional): Defaults to 768.
            maxPosEmbed (int, optional): Defaults to 50.
            sememeDim (int, optional): dimension of sememe embedding,here we use it pretrained in SE-WRL. Defaults to 768.
            attentionLayer (int, optional): count of attention layers in transformer decoder. Defaults to 8.
            head (int, optional): count of heads in one transformer layer. Defaults to 8.
            hiddenDim (int, optional): Defaults to 128.
            pretrained (bool, optional): if we use pretrained sememe embedding in SE-WRL. Defaults to True.
            TUPE (bool, optional): if we use TUPE mode, `False` means normal attention. Defaults to False.
            MASK (bool, optional): mask method allows us to only predict the existing father-child sememe pair appeared in train set. Defaults to True.
            seq (bool, optional): if we use seq, we will only use if NL-reader output as a vector, this is to compare with `no definition settings`. Defaults to False.
            depthMethod (str, optional): how to define the `position`. Defaults to 'none'.
                - none: means no position attention, only semantic attention.
                - forward: like normal transformer
                - depth: `tree position` 's define
            biasMethod (str, optional): 'distance' means. Defaults to 'none'.
        """        
        super(TreePred, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizer, BERT_PATH)    
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.dropout = nn.Dropout(p = 0.1)
        self.bert = model_class.from_pretrained(pretrained_weights)
        #for para in self.bert.parameters():
        #    para.requires_grad = False
        
        self.wordDim = wordDim
        self.sememeDim = sememeDim
        self.maxPosEmbed = maxPosEmbed
        self.attentionLayerC = attentionLayer
        self.hiddenDim = hiddenDim
        self.head = head
        self.tupe = TUPE
        self.mask = MASK
        self.seq = seq
        self.depthMethod = depthMethod
        self.biasMethod = biasMethod
        
        self.posEncoder = PositionalEncoding(d_model=sememeDim, dropout=0.1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #self.relativePositionEncoding = nn.Embedding(maxPosEmbed*maxPosEmbed*2 ,hiddenDim * attentionLayerC)
        #self.maxPosEmbed = maxPosEmbed
        
        self.biasWeight = torch.nn.Parameter(torch.Tensor(head,1000))
        nn.init.xavier_uniform_(self.biasWeight)
        
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
        
        
        self.candidateMask = torch.ones((len(self.id2sememe),len(self.id2sememe)),requires_grad = False)
        self.candidateMaskTemp = torch.ones((len(self.id2sememe),len(self.id2sememe)),requires_grad = False)
        
        if self.mask:
            # no mask means all postion is 1.0
            print("use mask method")
            self.createCandidate(SEMEME_PAIR_PATH)

        # last two represent start and end
        #self.sememeEmbedding = nn.Parameter(torch.rand((len(self.id2sememe),200), requires_grad = True))
        self.pretrained = pretrained
        if (pretrained == 1):
            self.sememeEmbedding = torch.rand((len(self.id2sememe),200))
        
            self.loadSememeEmbedding(SEMEME_EMBEDDING_PATH)
        else:
            self.sememeEmbedding = torch.rand((len(self.id2sememe),sememeDim))
        #self.sememeEmbedding.requires_grad = True
        self.sememeEmbedding = torch.nn.Parameter(self.sememeEmbedding)
        #usage: print(self.sememeEmbedding[self.sememe2id["Beethoven|贝多芬"]]) => tensor(200)
        
        self.embeddingChange = nn.Linear(200,sememeDim)
        
        
        #decoder_single = nn.TransformerDecoderLayer(d_model=wordDim, nhead=8)
        #self.decoder_layer = nn.TransformerDecoder(decoder_single, num_layers=6)
        
        # using my transformer here, remember to init
        c = copy.deepcopy
        feed_forward = PositionwiseFeedForward(d_model = wordDim, d_ff = 2048, dropout = 0.1)
        multi_head = MultiHeadedAttention(h = head,d_model=wordDim,dropout=0.1)
        self.multi_head_pos = MultiHeadedAttentionRel(h = head,d_model=wordDim,dropout=0.1)
        self.multi_head_pos_mem = MultiHeadedAttentionRel(h = head,d_model=wordDim,dropout=0.1)
        decoder_single = DecoderLayer(size=wordDim,self_attn = c(multi_head),src_attn= c(multi_head),feed_forward=c(feed_forward),dropout=0.1)
        self.decoder_layer = Decoder(decoder_single,self.attentionLayerC)
        for p in self.decoder_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        
        self.connect = nn.Linear(self.sememeDim,len(self.id2sememe))

        
        
        

    def createCandidate(self,candiFileName):
        """scan the candidate file, and generate a candidate mask, which enable the model to only predicti the existing sememe pairs in prediction settings.

        Args:
            candiFileName (string)
        """        
        # do a mask: 1 when i is j's candidate, else 0
        candi = torch.load(candiFileName)

        for id in tqdm.tqdm(range(len(self.id2sememe))):
            # for the ith sememe
            if id == len(self.candidateMask) - 1 or id == len(self.candidateMask) - 2:
                # "start" or "end"
                for sememeId in range(len(self.candidateMask)):
                    self.candidateMask[id][sememeId] = 1.0
            if candi.get(self.id2sememe[id],-1) != -1:
                for cont in candi[self.id2sememe[id]]:
                    #cont: [name,count]
                    self.candidateMask[id][self.sememe2id[cont[0]]] = 1.0
                    
            self.candidateMask[id][-1] = 1.0
            self.candidateMask[id][-2] = 0.0
        self.candidateMask[-2][-1] = 0.0
    
    def loadSememeEmbedding(self,sememeEmbeddingFileName):
        """load pretrained sememe embeddings

        Args:
            sememeEmbeddingFileName (string)
        """        
        # load pretrained sememeVecs, torch.rand() for except
        embeddings = open(sememeEmbeddingFileName,"r",encoding="utf-8").readlines()
        count = 0
        for contid in tqdm.tqdm(range(len(embeddings))):
            cont = embeddings[contid]
            name = cont.strip().split(" ")[0]
            vector = torch.tensor([float(i) for i in cont.strip().split(" ")[1:]] )
            for sememeId in range(len(self.id2sememe)):
                sememe = self.id2sememe[sememeId]
                if sememe.endswith(name):
                    count += 1
                    self.sememeEmbedding[sememeId] = vector
                    break
        print(f"{count} sememes find embeddings ")
    
    
    def saveBert(self, output_dir):
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(self.bert.state_dict(), output_model_file)
        self.bert.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(output_dir)
        
    def loadBert(self, output_dir):
        self.bert = tfs.BertModel.from_pretrained(output_dir)    
        self.tokenizer = tfs.BertTokenizer.from_pretrained(output_dir)
    
    
    def NLreader(self, batch_sentences):
        """follow code-generation's name, we use NL-reader to encode definition. 'SEQ' tells us if to use the sequence-like definition embedding

        Args:
            batch_sentences ([batch_size,string])

        Returns:
            torch_tensor[batch_size, bert_embedding_dim]
        """        
        # use BERT_BASE instead of transformer NLreader 
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                 pad_to_max_length=True)      #tokenize、add special token、pad
        #input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
        input_ids = torch.tensor(batch_tokenized['input_ids']).to(self.device)
        #print(input_ids)
        #attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(self.device)
        if self.seq:
            bert_output = self.bert(input_ids, attention_mask=attention_mask)[0][:,:,:]
        else:
            bert_output = self.bert(input_ids, attention_mask=attention_mask)[0][:,0:1,:]
        #bert_output = self.bert(input_ids, attention_mask=attention_mask)[1].unsqueeze(1)

        bert_output = self.dropout(bert_output)
        input_ids.to("cpu")
        attention_mask.to("cpu")
        return bert_output
    
    
    def dicts2tensor(self,dicts):
        """here we convert the sememes into real sememe embeddings in training process

        Args:
            dicts ([batch_size, sent_len]:string(name of sememe))

        Returns:
            dictsSeqVec (torch_tensor[batch_size,sent_len,sememe_embedding_dim])
            dictsSeqId (torch_tensor[batch_size,sent_len]:id of sememe) 
        """        
        assert(len(dicts[0]) == len(dicts[-1]))

        dictsSeqId = []
        dictsSeqVec = torch.zeros(len(dicts),len(dicts[0]),self.sememeDim)
        for contId in range(len(dicts)):
            dictsSeq = [self.sememe2id[i] for i in dicts[contId]]
            dictsSeqId.append(dictsSeq)
            #dictsSeq = [self.sememeEmbedding[i].cpu().detach().numpy() for i in dictsSeq]
            #dictsSeq = torch.tensor(dictsSeq).cuda()
            dictsSeq = [self.sememeEmbedding[i].cpu().detach().numpy() for i in dictsSeq]
            dictsSeq = torch.tensor(dictsSeq).to(self.device)
            if self.pretrained:
                dictsSeq = self.embeddingChange(dictsSeq)
            #size: [len,sememeDim]
            dictsSeqVec[contId] = dictsSeq
        
        return dictsSeqVec,dictsSeqId
    
    def computeDistance(self,stackList,id_other,id):
        dup = 0
        while dup < len(stackList[id]) and dup < len(stackList[id_other]) and stackList[id][dup] == stackList[id_other][dup]:
            dup += 1
        return len(stackList[id]) + len(stackList[id_other]) - 2 * dup
    
    def computeBiasAndDepth(self, dictsSeqId, mask = 0):
        length = len(dictsSeqId[0]) - mask # len of a tree seq
        bias = torch.zeros(len(dictsSeqId),self.head,length,length)
        depth = torch.zeros(len(dictsSeqId),length, dtype=torch.long)
        for treeSeqindex in range(len(dictsSeqId)):
            treeSeq = dictsSeqId[treeSeqindex]
            # for every seq in batch, init stack list and depthNow first
            stackList = [[] for _ in range(length)]
            depthNow = 0
            for id in range(length):
                # for each cont in seq
                if treeSeq[id] == 2083:
                    # st
                    stackList[id].append("st")
                elif treeSeq[id] == 2084:
                    # end
                    if stackList[id-1][-1][:2] == 'ba':
                        stackList[id] = stackList[id-1][:-2]
                        depthNow -= 2
                    else:
                        stackList[id] = stackList[id-1][:-1]
                        depthNow -= 1
                else:
                    if stackList[id-1][-1][:2] == 'ba':
                        stackList[id] = stackList[id-1][:-1] + ['ba_'+str(treeSeq[id]),str(treeSeq[id])]
                        depthNow += 1
                    else:
                        stackList[id] = stackList[id-1] + ['ba_'+str(treeSeq[id]),str(treeSeq[id])]
                        depthNow += 2
                #compute depth    
                if self.depthMethod == 'depth':
                    depth[treeSeqindex][id] = depthNow
                elif self.depthMethod == 'order':
                    depth[treeSeqindex][id] = id
                else:
                    print("we don't have this function")
                    exit()
                    
                    
                # compute distance
                bias[treeSeqindex,:,id,id] = self.biasWeight[:,0]
                for id_other in range(id):
                    dis = self.computeDistance(stackList,id_other,id)
                    bias[treeSeqindex,:,id,id_other] = self.biasWeight[:,dis]
                    bias[treeSeqindex,:,id_other,id] = self.biasWeight[:,dis]
                    
            # try [['2083','1','2','2084','3','4','2084','5','2084','2084','2084']] for example
                #print(stackList)
            #print(bias)
                #print(depth)
        return bias, depth
    
    def bias2weight(self, bias):
        """convert bias to weight embedding used in `tree-attention`

        Args:
            bias (torch_tensor[batch_size,sent_len,sent_len]:distance->int): the distance of every two nodes in a sememe tree

        Returns:
            weight: torch_tensor[batch_size,sent_len,bias_embedding_len]
        """        
        weight = torch.zeros(bias.shape[0],self.head,bias.shape[1],bias.shape[2])
        for batchId in range(bias.shape[0]):
            for id in range(bias.shape[1]):
                weight[batchId,:,id,id] = self.biasWeight[:,0]
                for id_other in range(bias.shape[2]):
                    weight[batchId,:,id,id_other] = self.biasWeight[:,bias[batchId,id,id_other]]
                    weight[batchId,:,id_other,id] = self.biasWeight[:,bias[batchId,id_other, id]]
        return weight
       

    def forward(self, NL, dicts , bias = None, depth = None):
        """implement of transformer. support normal mode, TUPE mode, tree-attention mode

        Args:
            NL ([batch_size]->string): the defination of synset.
            dicts ([batch_size,sent_len - 1]->string(tree sequence)): defined in `self.dicts2tensor`.
            bias :Defaults to None.
            depth :Defaults to None.

        Returns:
            out ([batch_size, sent_len - 1, sememe_count]): distribution in position[1:], result to compute loss in train mode
            dictsSeqId: defined in `self.dicts2tensor`
        """        
        # input: n sentences
        NLoutput = self.NLreader(NL)
        #size-bert-CLS : [batchsize, 1, wordDim]
        dictsSeqVec,dictsSeqId = self.dicts2tensor(dicts)

        #dictsSeqVec = dictsSeqVec.cuda()
        dictsSeqVec = dictsSeqVec.to(self.device)
        if self.tupe:
            dictsSeqVec = dictsSeqVec[:,:-1,:]
            # size: [batchSize, tgtLen - 1, wordDim]
            if bias == None:
                bias,depth = self.computeBiasAndDepth(dictsSeqId, mask = 1)
            #bias = self.bias2weight(bias)
            tgt_mask = make_std_mask(dictsSeqVec[:,:,0],0)
            if self.biasMethod != 'none':
                attn_pos = self.multi_head_pos(query=dictsSeqVec,key=dictsSeqVec,value=dictsSeqVec,mask = tgt_mask, postionMat = depth, bias = bias, device=self.device)
            else:
                attn_pos = self.multi_head_pos(query=dictsSeqVec,key=dictsSeqVec,value=dictsSeqVec,mask = tgt_mask, postionMat = depth, device=self.device)                
            #attn_pos_mem = self.multi_head_pos(query=dictsSeqVec,key=NLoutput,value=NLoutput)
            out = self.decoder_layer(x = dictsSeqVec, memory = NLoutput, tgt_mask = tgt_mask, src_mask = None, attn_pos= attn_pos)
        else:
            dictsSeqVec = self.posEncoder(dictsSeqVec)
            dictsSeqVec = dictsSeqVec[:,:-1,:]
            # size: [batchSize, tgtLen - 1, wordDim]
            
            tgt_mask = make_std_mask(dictsSeqVec[:,:,0],0)
            out = self.decoder_layer(x = dictsSeqVec, memory = NLoutput, tgt_mask = tgt_mask, src_mask = None)            
            
        out = self.connect(out)
        # out: [batchSize, tgtLen - 1, sememeCount]
        dictsSeqVec.to("cpu")
        NLoutput.to("cpu")         
        tgt_mask.to("cpu")   
        return out , dictsSeqId
        
        
    def visualize(self, NL, seq):
        """to vistualize tree attention score for input, used in `analysis`, you can ignore it.
        """        
        NLoutput = self.NLreader(NL)
        #size-bert-CLS : [batchsize, 1, wordDim]
        dictsSeqVec,dictsSeqId = self.dicts2tensor([seq])
        #print(dictsSeqVec.size())
        #print(dictsSeqId)
        bias,depth = self.computeBiasAndDepth(dictsSeqId, mask = 0)
        #print(depth)
        attn_pos = self.multi_head_pos(query=dictsSeqVec,key=dictsSeqVec,value=dictsSeqVec,postionMat = depth, bias = bias, device=self.device)
        attn_pos = torch.mean(attn_pos, dim = 1)

        attn_pos = torch.squeeze(attn_pos, dim= 0)
        out = self.decoder_layer(x = dictsSeqVec, memory = NLoutput, tgt_mask = None, src_mask = None, attn_pos= attn_pos)
        global v_o
        v_o = torch.mean(v_o, dim = 1)
        v_o = torch.squeeze(v_o, dim = 0)
        semantic_attn = v_o - attn_pos
        
        return semantic_attn, attn_pos, v_o

    def decodeList(self,NLout,idList, beamWidth):
        """implement beam search method for prediction

        Args:
            NLout : encoding result of NL-reader
            idList : _description_
            beamWidth : search width for beam serach

        Returns:
            result: prediciton result
        """        
        #temp = torch.tensor([self.sememeEmbedding[item].cpu().detach().numpy() for item in idList[0]]).cuda() 
        temp = torch.tensor([self.sememeEmbedding[item].cpu().detach().numpy() for item in idList[0]]).to(self.device)
        if self.pretrained:
            temp = self.embeddingChange(temp)
        temp = temp.unsqueeze(0) # make batch_size = 1
        
        if self.tupe:
            bias,depth = self.computeBiasAndDepth([idList[0]])
            #bias = self.bias2weight(bias)
            if self.biasMethod != "none":
                attn_pos = self.multi_head_pos(query=temp,key=temp,value=temp,bias=bias,postionMat=depth,device=self.device)
            else:
                attn_pos = self.multi_head_pos(query=temp,key=temp,value=temp, postionMat=depth,device=self.device)
            #attn_pos_mem = self.multi_head_pos(query=temp,key=NLout,value=NLout)
            out = self.decoder_layer(x = temp, memory = NLout,tgt_mask = None, src_mask = None, attn_pos=attn_pos)
        else:
            temp = self.posEncoder(temp)
            out = self.decoder_layer(x = temp, memory = NLout,tgt_mask = None, src_mask = None)           
        # out: [1 , sentenceLen, wordDim]
        pred = self.connect(out[-1,-1,:])

        temp.to("cpu")
        for id in range(len(self.id2sememe)):
                if self.candidateMaskTemp[idList[0][-1]][id] == 0.0:
                    pred[id] = -float("inf")
        
        pred = nn.Softmax(dim=0)(pred) 
        values,indices = torch.topk(pred,k = beamWidth, dim = 0)

        result = []
        for id in range(len(values)):
            depth = idList[2]
            if (int(indices[id]) == self.sememe2id['end']):
                depth -= 1
            else:
                depth += 1
                # not 'end', don't use indices[id] in 
                self.candidateMaskTemp[:,int(indices[id])] = torch.zeros(len(self.sememe2id))
            result.append([ idList[0]+[int(indices[id])], float(idList[1]*values[id]), depth])
        pred.to("cpu")
        return result
    
                    
    
    def predict(self, NL, maxLen=15, beamWidth = 1, stricted = None):
        """the prediciton method for the model

        Args:
            NL ([batch_size]:string): definitions
            maxLen (int, optional): the max len of predicted sememe tree, this work because every prefix can be decoded into a sememe tree(though it may not ended). Defaults to 15.
            beamWidth (int, optional): search width. Defaults to 1.
            stricted (bool, optional): whether we can predict the same sememe in one run. Defaults to None.

        Returns:
            [batch_size,[tree sequence]]: prediction result
        """        
        # input: n sentences
        
        #NLoutput = self.NLreader(NL).cuda()
        NLoutput = self.NLreader(NL).to(self.device)

        # make candidateMaskTemp same as candidateMask
        if stricted != None:
            self.candidateMaskTemp = torch.zeros([len(self.id2sememe),len(self.id2sememe)]).to(self.device)
            for sememe in stricted[1:]:
                self.candidateMaskTemp[:,self.sememe2id[sememe]] = 1.0
        else:
            self.candidateMaskTemp = self.candidateMask.clone()

        cands = []
        cands.append([[self.sememe2id['start']],1.0,1])
        # cand = [[content list] , posibility, depth ]
        length = 1
        finished = 0
        
        while (length < maxLen and finished == 0):
            newCand = []
            
            for cont in cands:
                #print(cont)
                if cont[2] == 0:
                    # depth=0, this tree is finished
                    newCand.append(cont)
                else:
                    # try all the posibilities
                    #if length == 1:
                    #    newCand = newCand + self.decodeList(NLoutput,cont,10*beamWidth)
                    #else:
                    newCand = newCand + self.decodeList(NLoutput,cont,beamWidth)
            
            # sort the result and take first k(beamWidth)
            newCand.sort(key=lambda x : x[1],reverse=True)
            if len(newCand) > beamWidth:
                cands = newCand[:beamWidth]
            else:
                cands = newCand
            
            finished = 1
            for cont in cands:
                # if all cand in cands have finished
                if cont[2] != 0:
                    finished = 0
                    break
            length += 1
        #print(cands)
        NLoutput.to("cpu")
        # return the biggest
        return cands[0][0]

    



if __name__ == "__main__":
    pass