# -*- coding: utf-8 -*-
# file: train.py
# author: Liwenwen <lww503@126.com>

import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy
import torch.nn.functional as F
import json
import pickle
from transformers import BertModel

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer,  Tokenizer4Bert, ABSADataset

from models import LA_GCN


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def read_sentence_depparsed(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data

def get_graph_dataset(dataset_name):
    rest_train = ' '
    rest_test = ' '

    laptop_train = ' '
    laptop_test = ' '

    twitter_train = ' '
    twitter_test = ' '

    ds_train = {'restaurant': rest_train,
                'laptop': laptop_train, 'twitter': twitter_train}
    ds_test = {'restaurant': rest_test,
               'laptop': laptop_test, 'twitter': twitter_test}

    fin = open(ds_train[dataset_name], 'rb')
    train_graph = pickle.load(fin)
    fin.close()

    fin = open(ds_test[dataset_name], 'rb')
    test_graph = pickle.load(fin)
    fin.close()
    return train_graph, test_graph


def get_dep_dataset(dataset_name):
    rest_train = ' '
    rest_test = ' '

    laptop_train = ' '
    laptop_test = ' '

    twitter_train = ' '
    twitter_test = ' '

    ds_train = {'restaurant': rest_train,
                'laptop': laptop_train, 'twitter': twitter_train}
    ds_test = {'restaurant': rest_test,
               'laptop': laptop_test, 'twitter': twitter_test}

    train = list(read_sentence_depparsed(ds_train[dataset_name]))
    logger.info('# Read %s Train set: %d', dataset_name, len(train))

    test = list(read_sentence_depparsed(ds_test[dataset_name]))
    logger.info("# Read %s Test set: %d", dataset_name, len(test))
    return train, test

class Instructor:
    def __init__(self, opt):
        self.opt = opt

        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)

        train_dep, test_dep = get_dep_dataset(self.opt.dataset)           # get dependency information
        train_graph,test_graph = get_graph_dataset(self.opt.dataset)       # get the dependency matrix
        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer,self.opt,train_dep,train_graph)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer,self.opt,test_dep,test_graph)
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0

        global_step = 0
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('Epoch[{}/{}'.format(epoch + 1, self.opt.num_epoch))
            # logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)

                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                val_acc, val_f1= self._evaluate_acc_f1(val_data_loader)
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    if not os.path.exists('state_dict'):
                        os.mkdir('state_dict')
                    path = 'state_dict/{0}_{1}_val_acc{2}'.format(self.opt.model_name, self.opt.dataset,
                                                                  round(val_acc, 4))
                    torch.save(self.model.state_dict(), path)
                    logger.info('>> saved: {}'.format(path))
                if val_f1 > max_val_f1:
                    max_val_f1 = val_f1
                logger.info(
                    '[step {:.4f}], train_loss: {:.4f}, train_acc: {:.4f}, val_acc: {:.4f}, val_f1: {:.4f}'.format(
                        global_step, train_loss, train_acc, val_acc, val_f1))
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(max_val_acc, max_val_f1))
        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        n_loss = 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                loss = F.cross_entropy(t_outputs, t_targets)
                n_loss = n_loss + loss

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        los = n_loss / len(data_loader)
        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='la_gcn', type=str)
    parser.add_argument('--dataset', default='restaurant', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)  # 优化器
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--hidden_dim', default=768, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--num_heads', type=int, default=6, help='Number of heads for gat.')
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--multi_hop', type=bool, default=True,
                        help='Multi hop non connection.')
    parser.add_argument('--max_hop', type=int, default=4,
                        help='max number of hops')
    parser.add_argument('--embedding_type', type=str, default='bert')
    parser.add_argument('--pure_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'la_gcn': LA_GCN,
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }
    input_colses = {
        'la_gcn' :['text_bert_indices', 'bert_segments_ids','text_raw_bert_indices', 'aspect_bert_indices','dependency_graph','tags']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  
        'adagrad': torch.optim.Adagrad,  
        'adam': torch.optim.Adam, 
        'adamax': torch.optim.Adamax,  
        'asgd': torch.optim.ASGD,  
        'rmsprop': torch.optim.RMSprop,  
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]    
    opt.dataset_file = dataset_files[opt.dataset]       
    opt.inputs_cols = input_colses[opt.model_name]      
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
