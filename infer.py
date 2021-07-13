# -*- coding: utf-8 -*-
# file: infer.py
# author: Liwenwen <lww503@126.com>
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertModel
from data_utils import Tokenizer4Bert
import argparse
import torch.nn as nn
import os
from models.la_gcn import LA_GCN
from allennlp.predictors.predictor import Predictor
from collections import Counter, defaultdict
from transformers import BertModel, BertConfig
from pytorch_pretrained_bert import BertTokenizer
import re

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

from ddparser import DDParser
import json
import jieba
import jieba.posseg as pseg
sentiment_map = {0: 'neutral', 1: 'positive', -1: 'negative'}
jieba.enable_paddle()
important_tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
logger = logging.getLogger(__name__)



def syntaxInfo2json(sentences, sentences_with_dep):
    num_aspect =0
    nn_tag = ['NN', 'NNS', 'NNP', 'NNPS']
    sentences['tokens'] = sentences_with_dep['words']
    sentences['tags'] = sentences_with_dep['pos']
    tags = sentences['tags']
    for tag in tags:
        if tag in nn_tag:
            num_aspect = num_aspect + 1
    sentences['num_aspect'] = num_aspect
    sentences['predicted_dependencies'] = sentences_with_dep['predicted_dependencies']
    sentences['dependencies'] = []
    sentences['predicted_heads'] = sentences_with_dep['predicted_heads']
    predicted_heads = sentences_with_dep['predicted_heads']
    predicted_dependencies = sentences_with_dep['predicted_dependencies']
    for idx, item in enumerate(predicted_dependencies):
        dep_tag = item
        frm = predicted_heads[idx]
        to = idx + 1
        sentences['dependencies'].append([dep_tag, frm, to])
    dependences = sentences['dependencies']
    for dep in dependences:
        if dep[1] == 0:
            sentences['root'] = dep[2]
            break
    return sentences

def deteal_root_dep(input_data):
    all_data = []
    e=input_data
    dependences = e['dependencies']
    max_distance = 0
    if e['num_aspect']<2:
        new_dependencies = dependences
    else:
        new_dependencies = []
        for dep in dependences:
            if dep[1] == int(e['root']):
                distance = abs(dep[2]-dep[1])
                if distance > max_distance and dep[0]!='punct':
                    max_distance = distance
                    node = dep[2]
        for dep in dependences:
            if dep[2]!= node:
                dep_tag = dep[0]
                frm = dep[1]
                to = dep[2]
                new_dependencies.append([dep_tag, frm, to])
            else:
                dep_tag = dep[0]
                frm = 0
                to = dep[2]
                new_dependencies.append([dep_tag, frm, to])
    all_data.append(
        {'sentence': e['sentence'],'from_to': e['from_to'], 'tokens': e['tokens'], 'tags': e['tags'], 'predicted_dependencies': e['predicted_dependencies'], 'dependencies': e['dependencies'], 'predicted_heads': e['predicted_heads'],'root':e['root'],'num_aspect':e['num_aspect'],'new_dependencies':new_dependencies})
    return all_data


def reshape_dependency_tree_new(as_start, as_end, dependencies,tags, multi_hop=False, tokens=None, max_hop = 5):
    dep_tag = []
    dep_idx = []
    for i in range(as_start, as_end):
        for dep in dependencies:
            if i == dep[1] - 1:
                if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':
                        dep_tag.append(dep[0])
                    else:
                        dep_tag.append('<pad>')
                    dep_idx.append(dep[2] - 1)
            elif i == dep[2] - 1:
                if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':
                        dep_tag.append(dep[0])
                    else:
                        dep_tag.append('<pad>')
                    dep_idx.append(dep[1] - 1)

    if multi_hop:
        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(dep_idx)
            for i in dep_idx_temp:
                for dep in dependencies:
                    if i == dep[1] - 1:
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':
                                dep_tag.append(dep[0])
                            else:
                                dep_tag.append('<pad>')
                            dep_idx.append(dep[2] - 1)
                            added = True
                    elif i == dep[2] - 1:
                        # not root, not aspect
                        if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':
                                dep_tag.append(dep[0])
                            else:
                                dep_tag.append('<pad>')
                            dep_idx.append(dep[1] - 1)
                            added = True
            current_hop += 1

    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x:x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]

    return dep_tag, dep_idx



def get_rolled_and_unrolled_data(input_data, args):
    all_rolled = []
    total_counter = defaultdict(int)
    mixed_counter = defaultdict(int)
    sentiments_lookup = {'negative': 0, 'positive': 1, 'neutral': 2}
    important_tags = ['JJ','JJR','JJS','RB','RBR','RBS','MD','VB','VBD','VBG','VBN','VBP','VBZ']

    logger.info('*** Start processing data ***')
    for e in input_data:
        e['tokens'] = [x.lower() for x in e['tokens']]
        froms = []
        tos = []
        dep_tags = []
        dep_index = []

        pos_class = e['tags']
        for i in range(1):
            frm = e['from_to'][i][0]
            to = e['from_to'][i][1]

            froms.append(frm)
            tos.append(to)

            dep_tag,dep_idx = reshape_dependency_tree_new(frm, to, e['new_dependencies'],e['tags'],
                                                       multi_hop=args.multi_hop, tokens=e['tokens'], max_hop=args.max_hop)

            dep_tags.append(dep_tag)
            dep_index.append(dep_idx)
            new_tags = []
            for i in range(len(pos_class)):
                if i in dep_index[0]:
                    new_tags.append(pos_class[i])
                else:
                    new_tags.append('non')
        all_rolled.append(
            {'sentence': e['tokens'], 'tags': e['tags'],'from': froms, 'to': tos, 'dep_tags': dep_tags, 'dep_ids': dep_index, 'dependencies': e['dependencies'], 'new_tags':new_tags,'new_dependencies':e['new_dependencies']})

    logger.info('Total sentiment counter: %s', total_counter)
    logger.info('Multi-Aspect-Multi-Sentiment counter: %s', mixed_counter)

    return all_rolled

def get_graph(train_dep):
    idx2graph = {}
    m = 0
    for dep in train_dep:
        len_text = len(dep['sentence'])
        dependencies = dep['new_dependencies']

        matrix = np.zeros((80, 80)).astype('float32')
        for x in range(1, len_text + 1):
            if x <= 79:
                matrix[x][x] = 1
        for new_dep in dependencies:
            if new_dep[1] <= 79 and new_dep[2] <= 79:
                matrix[new_dep[1]][new_dep[2]] = 1
                matrix[new_dep[2]][new_dep[1]] = 1
                continue
        idx2graph[m] = matrix
        m = m + 1
    return idx2graph

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

def get_sentence(text_left, aspect, text_right):
    sentence = dict()
    sentence['sentence'] = text_left + " " + aspect + " " + text_right
    frm = len(text_left.split())+1
    to = frm + len(aspect.split())
    sentence['from_to'] = [[frm, to]]
    return sentence


def prepare_data(text_left, aspect, text_right, tokenizer,opt):
    model_path = os.path.join("./datasets/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
    text_left = text_left.lower().strip()
    text_right = text_right.lower().strip()
    aspect = aspect.lower().strip()
    predictor = Predictor.from_path(model_path)
    text = text_left + " " + aspect + " " + text_right
    pre = predictor.predict(sentence=text)
    sentence = get_sentence(text_left, aspect, text_right)
    depen = syntaxInfo2json(sentence, pre)
    train_root = deteal_root_dep(depen)
    train_dep = get_rolled_and_unrolled_data(train_root, opt)
    dependency_graph = get_graph(train_dep)

    text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
    aspect_indices = tokenizer.text_to_sequence(aspect)
    aspect_len = np.sum(aspect_indices != 0)
    text_bert_indices = tokenizer.text_to_sequence(
        '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
    text_raw_bert_indices = tokenizer.text_to_sequence(
        "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
    bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
    bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)
    aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

    left_context_len = train_dep[0]['from'][0]
    aspect_len = np.sum(aspect_indices != 0)
    tags_ids = []
    num = 0
    for i in range(len(train_dep[0]['new_tags'])):
        tags = train_dep[0]['new_tags']
        num = num + 1
        if i <= opt.max_seq_len - 2:
            if tags[i] in important_tags:
                tags_ids.append(1)
            elif left_context_len <= i and i < left_context_len + aspect_len:
                tags_ids.append(1)
            elif tags[i] == 'non':
                tags_ids.append(0)
            else:
                tags_ids.append(2)
    if num <= 79:
        all_tag_ids = np.asarray([0] + tags_ids + [0] * (opt.max_seq_len - len(tags_ids) - 1))
    else:
        all_tag_ids = np.asarray([0] + tags_ids)

    return text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices, dependency_graph, all_tag_ids


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='la_gcn', type=str)
    parser.add_argument('--dataset', default='restaurant', type=str, help='twitter, restaurant, laptop') 
    parser.add_argument('--optimizer', default='adam', type=str)  
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--hidden_dim', default=768, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--multi_hop', type=bool, default=True,
                        help='Multi hop non connection.')
    parser.add_argument('--max_hop', type=int, default=4,
                        help='max number of hops')
    parser.add_argument('--embedding_type', type=str, default='bert', choices=['glove', 'bert'])
    parser.add_argument('--pure_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    model_classes = {
        'la_gcn': LA_GCN
    }
    # set your trained models here
    state_dict_paths = {
        'la_gcn': ' ',
    }

    opt = get_parameters()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
    bert = BertModel.from_pretrained(opt.pretrained_bert_name)
    model = model_classes[opt.model_name](bert, opt).to(opt.device)

    print('loading model {0} ...'.format(opt.model_name))
    model.load_state_dict(torch.load(state_dict_paths[opt.model_name]))
    model.eval()
    torch.autograd.set_grad_enabled(False)

    text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices, dependency_graph,tags= \
        prepare_data('This little place has a cute', 'interior decor', 'and affordable city prices.',tokenizer,opt)

    text_bert_indices = torch.tensor([text_bert_indices], dtype=torch.int64).to(opt.device)
    bert_segments_ids = torch.tensor([bert_segments_ids], dtype=torch.int64).to(opt.device)
    text_raw_bert_indices = torch.tensor([text_raw_bert_indices], dtype=torch.int64).to(opt.device)
    aspect_bert_indices = torch.tensor([aspect_bert_indices], dtype=torch.int64).to(opt.device)
    dependency_graph = torch.tensor([dependency_graph[0]], dtype=torch.int64).to(opt.device)
    tags = torch.tensor([tags], dtype=torch.int64).to(opt.device)


    if 'la_gcn' in opt.model_name:
        inputs = [text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices,dependency_graph,tags]
    outputs = model(inputs)
    t_probs = F.softmax(outputs, dim=-1).cpu().numpy()
    print('t_probs = ', t_probs)
    print('aspect sentiment = ', t_probs.argmax(axis=-1) - 1)

