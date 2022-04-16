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

import sys
import numpy as np
#import preprocessor
from model import TreePred
from train import testFirst


TEST_NAME = ""
MODEL_PATH = ""

'''
model settings must be the same as the training settings
'''
model = TreePred(wordDim = 768, maxPosEmbed = 50,sememeDim = 768, head=8,attentionLayer = 8,hiddenDim = 128, pretrained = 1, TUPE=1, MASK=1, seq = False, depthMethod="depth", biasMethod="distance")


stateTest = torch.load(MODEL_PATH+"/modelWeight.pt",map_location=torch.device('cpu'))
model.load_state_dict(stateTest['net'],strict=False)
model.loadBert(MODEL_PATH+"/")
model.device = torch.device("cpu")
model.to('cpu')
model.eval()

with torch.no_grad():
    results = testFirst(model,TEST_NAME)
    print(results)
