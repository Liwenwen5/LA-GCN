# -*- coding: utf-8 -*-

import copy
import numpy as np
from layers.attention import Attention

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from transformers.modeling_bert import BertPooler, BertSelfAttention


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
                                            dtype=np.float32), dtype=torch.float32).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class LA_GCN(nn.Module):
    def __init__(self, bert, opt):
        super(LA_GCN, self).__init__()
        self.bert = bert
        self.bert_local = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.gc = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.onedense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.densee = nn.Linear(opt.bert_dim * 2, opt.polarities_dim)
        self.linear_double = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.linear_trouble = nn.Linear(opt.bert_dim * 3, opt.bert_dim)
        self.attn = Attention(opt.bert_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)

    def tag_mask(self, text_bert_indices, tags):
        tags = tags.cpu().numpy()
        texts = text_bert_indices.cpu().numpy()
        masked_text_raw_indices = np.ones((text_bert_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i in range(len(texts)):
            distances = tags[text_i]
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def weighted_mask(self, text_local_indices, aspect_indices, tags):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        tags = tags.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(1, np.count_nonzero(texts[text_i]) - 1):
                if i >= asp_begin and i < asp_begin + asp_len:
                    distances[i] = 1
                elif tags[text_i][i] == 0.2:
                    distances[i] = 1 - (abs(i - asp_avg_index) + asp_len / 2) / np.count_nonzero(texts[text_i])
                elif tags[text_i][i] == 1:
                    distances[i] = 1
                else:
                    distances[i] = 0
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]
        adj = inputs[4]
        tags = inputs[5]

        bert_out, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)  # torch.Size([16, 80, 768])
        bert_out = self.dropout(bert_out)

        bert_local_out, _ = self.bert_local(text_local_indices)
        bert_local_out = self.dropout(bert_local_out)

        tag_mask = self.weighted_mask(text_local_indices,aspect_indices,tags)
        tag_mask_out = torch.mul(bert_local_out, tag_mask)

        gcn_out = F.relu(self.gc(tag_mask_out, adj))
        gcn_out = self.dropout(gcn_out)

        out_cat = torch.cat((gcn_out, bert_out), dim=-1)
        mean_pool = self.linear_double(out_cat)

        self_attention_out = self.bert_SA(mean_pool)
        all_out = torch.div(torch.sum(self_attention_out, dim=1), self.opt.max_seq_len)

        dense_out = self.onedense(all_out)

        return dense_out