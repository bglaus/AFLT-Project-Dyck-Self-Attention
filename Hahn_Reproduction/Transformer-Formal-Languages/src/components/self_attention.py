import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
from torch.autograd import Variable
import ipdb as pdb
from src.components.utils import *

def hardmax(scores):
    idx = torch.argmax(scores, dim=-1, keepdims=True)
    ret = torch.zeros_like(scores).scatter_(-1, idx, 1.)
    return ret

def attention(query, key, value, mask=None, dropout=None, a_type='soft'):
	"Compute 'Scaled Dot Product Attention'"
	#pdb.set_trace()
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) \
			 / math.sqrt(d_k)
	if mask is not None:
		scores = scores + mask
	if a_type == 'soft': 
        # soft-attention: use softmax of scores
        p_attn = F.softmax(scores, dim = -1)
	else: 
        # hard-attention: use actual maximum attention values
        # description of hardmax function in https://arxiv.org/pdf/1901.03429.pdf
        p_attn = hardmax(scores)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1, bias = True,
				freeze_q = False, freeze_k = False,
				freeze_v = False, zero_k = False):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		self.bias = bias
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model, bias = bias), 4)
		if freeze_q:
			self.linears[0].requires_grad_(False)
		if freeze_k:
			self.linears[1].requires_grad_(False)
		if freeze_v:
			self.linears[2].requires_grad_(False)
		if zero_k:
			self.null_linear_layer(self.linears[1])
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)

	def null_linear_layer(self, ln):
		with torch.no_grad():
			ln.weight.fill_(0.0)
			if self.bias:
				ln.bias.fill_(0.0)
		ln.requires_grad_(False)
		
	def forward(self, query, key, value, mask=None, a_type='soft'):
		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(0).unsqueeze(0)
		nbatches = query.size(0)
		
		# 1) Do all the linear projections in batch from d_model => h x d_k 
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.linears, (query, key, value))]
		
		# 2) Apply attention on all the projected vectors in batch. 
		x, self.attn = attention(query, key, value, mask=mask, 
								 dropout=self.dropout, a_type=a_type)
		
		# 3) "Concat" using a view and apply a final linear. 
		x = x.transpose(1, 2).contiguous() \
			 .view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)
