# -*- coding: utf-8 -*-
# file: data_utils.py
# author: Liwenwen <lww503@126.com>

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from collections import Counter, defaultdict



import json
from torch.nn.utils.rnn import pad_sequence
import argparse
import codecs
import json
import linecache
import logging
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from copy import copy, deepcopy

import nltk
import simplejson as json
import torch
from allennlp.modules.elmo import batch_to_ids
from lxml import etree
from nltk import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from torch.utils.data import DataLoader, Dataset
logger = logging.getLogger(__name__)

def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


    def read_sentence_depparsed(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data

    def get_dep_dataset(self,dataset_name):
        rest_train = ''
        rest_test = ''

        laptop_train = ''
        laptop_test = ''

        twitter_train = ''
        twitter_test = ''

        ds_train = {'restaurant': rest_train,
                    'laptop': laptop_train, 'twitter': twitter_train}
        ds_test = {'restaurant': rest_test,
                   'laptop': laptop_test, 'twitter': twitter_test}

        train = list(read_sentence_depparsed(ds_train[dataset_name]))
        logger.info('# Read %s Train set: %d', dataset_name, len(train))

        test = list(read_sentence_depparsed(ds_test[dataset_name]))
        logger.info("# Read %s Test set: %d", dataset_name, len(test))
        return train_dep, test_dep


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer,opt,dep,idx2gragh):
        important_tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()     
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):    
            x=i//3
            tags = dep[x]['new_tags']
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]  
            aspect = lines[i + 1].lower().strip()          
            polarity = lines[i + 2].strip()                
            dependency_graph = idx2gragh[x]

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)   
            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)              
            polarity = int(polarity) + 1

            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")   
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))    
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)      

            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")
            tags_ids =[]

            text_left_indices = tokenizer.text_to_sequence(text_left)
            left_context_len = np.sum(text_left_indices != 0)  
            
            for i in range(len(tags)):
                if tags[i] in important_tags:
                    tags_ids.append(1)
                elif left_context_len <= i and i < left_context_len+aspect_len :
                    tags_ids.append(1)
                elif tags[i]=='non':
                    tags_ids.append(0)
                else:
                    tags_ids.append(0.2) 
            all_tag_ids = np.asarray([0]+ tags_ids + [0] * (tokenizer.max_seq_len - len(tags_ids)-1))
            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'dependency_graph': dependency_graph,
                'tags': all_tag_ids,
                'polarity': polarity,
            }
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
