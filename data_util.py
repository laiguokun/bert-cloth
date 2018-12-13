import os, sys
import glob
import time

import numpy as np
import torch
import json
import nltk
import argparse
import fnmatch
import random

from pytorch_pretrained_bert.tokenization import BertTokenizer

def get_json_file_list(data_dir):
    files = []
    for root, dir_names, file_names in os.walk(data_dir):
        for filename in fnmatch.filter(file_names, '*.json'):
            files.append(os.path.join(root, filename))
    return files

def tokenize_ops(ops, tokenizer):
    ret = []
    for i in range(4):
        ret.append(tokenizer.tokenize(ops[i]))
    return ret

def to_device(L, device):
    if (type(L) != list):
        return L.to(device)
    else:
        ret = []
        for item in L:
            ret.append(to_device(item, device))
        return ret

class ClothSample(object):
    def __init__(self, data, tokenizer):
        self.article = None
        self.ph = []
        self.ops = []
        self.ans = []
        if (data != None):
            cnt = 0
            self.article = tokenizer(self.article)
            for p in range(len(self.article)):
                if (self.article[p] == '_'):
                    self.article[p] = '[MASK]'
                    self.ph.append(p)
                    ops = tokenize_ops(data['option'][cnt], self.tokenizer)
                    self.ops.append(ops)
                    self.ans.append(ord(data['answer'][cnt]) - ord('A'))
                    cnt += 1
                    
    def convert_tokens_to_ids(self, tokenizer):
        self.article = tokenizer.convert_tokens_to_ids(self.article)
        self.article = torch.Tensor(self.article)
        for i in range(len(self.ops)):
            for k in range(4):
                self.ops[i][k] = tokenizer.convert_tokens_to_ids(self.ops[i][k])
                self.ops[i][k] = torch.tensor(self.article)
        self.ph = torch.Tensor(self.ph)
        self.ans = torch.Tensor(self.ans)
                
        
class Preprocessor(object):
    def __init__(self, args, device='cpu'):
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        self.data_dir = args.data_dir
        file_list = get_json_file_list(args.data_dir)
        self.data = []
        #max_article_len = 0
        for file_name in file_list:
            data = json.loads(open(file_name, 'r').read())
            self.data.append(data)
            #max_article_len = max(max_article_len, len(nltk.word_tokenize(data['article'])))
        self.data_split = []
        for sample in self.data:
            self.data_split += self._split_data(sample, args.pre, args.post)
            #break
        
        for i in range(len(self.data_split)):
            self.data_split[i].convert_tokens_to_ids(self.tokenizer)
            #break
        torch.save(self.data_split, args.save_name)
        
        
    
    def _split_data(self, data, pre=1, post=1):
        if (pre + post == 0):
            return [self._create_sample(self.data)]
        ret = []
        sents = nltk.sent_tokenize(data['article'])
        for i in range(len(sents)):
            sents[i] = self.tokenizer.tokenize(sents[i])
            for p in range(len(sents[i])):
                if (sents[i][p] == '_'):
                    sents[i][p] = '[MASK]'
        cnt = 0
        for i in range(len(sents)):
            sample = self._create_sample()
            article = []
            for left in range(i-1, max(0, i - pre - 1), -1):
                article += sents[left]
            article += sents[i]
            for right in range(i+1, min(i + post + 1, len(sents))):
                article += sents[right]
            sample.article = article
            
            for p in range(len(sents[i])):
                if (sents[i][p] == '[MASK]'):
                    sample.ph.append(p)
                    ops = tokenize_ops(data['options'][cnt], self.tokenizer)
                    sample.ops.append(ops)
                    sample.ans.append(ord(data['answers'][cnt]) - ord('A'))
                    cnt += 1
            if (len(sample.ph) > 0):
                ret.append(sample)
        return ret
    
    def _create_sample(self, data=None):
        return ClothSample(data, self.tokenizer)

class Loader(object):
    def __init__(self, data_dir, data_file, cache_size, batch_size, device='cpu'):
        #self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        self.data_dir = os.path.join(data_dir, data_file)
        print('loading {}'.format(self.data_dir))
        self.data = torch.load(self.data_dir)
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.data_num = len(self.data)
        self.device = device
    
    def _batchify(self, data_set, data_batch):
        articles = []
        articles_mask = []
        options = []
        options_mask = []
        answers = []
        question_id = []
        question_pos = []
        
        max_article_length = 0
        max_option_length = 0
        for idx in data_batch:
            data = data_set[idx]
            max_article_length = max(max_article_length, data.article.size(0))
            for ops in data.ops:
                for op in ops:
                    max_option_length = max(max_option_length, op.size(0))
                    
        for i, idx in enumerate(data_batch):
            data = data_set[idx]
            padding = torch.zeros(max_article_length - data.article.size(0))
            article = torch.cat([data.article, padding], 0)
            mask = torch.zeros(max_article_length)
            mask[:data.article.size(0)] = 1
            articles.append(article)
            articles_mask.append(mask)
            for ops in data.ops:
                o = []
                m = []
                for op in ops:
                    padding = torch.zeros(max_option_length - op.size(0))
                    o.append(torch.cat([op, padding], 0))
                    mask = torch.zeros(max_option_length)
                    mask[:op.size(0)] = 1
                    m.append(mask)
                o = torch.stack(o, 0)
                m = torch.stack(m, 0)
                options.append(o)
                options_mask.append(m)
                question_id.append(i)
            for ans in data.ans:
                answers.append(ans)
            for pos in data.ph:
                question_pos.append(pos + i * max_article_length)
        articles = torch.stack(articles).long() #bsz X alen
        articles_mask = torch.stack(articles_mask)
        options = torch.stack(options).long() #opnum X 4 X oplen
        options_mask = torch.stack(options_mask)
        question_id = torch.LongTensor(question_id) #opnum
        question_pos = torch.LongTensor(question_pos) #opnum
        answers = torch.LongTensor(answers) #opnum
        inp = [articles, articles_mask, options, options_mask,
               question_id, question_pos]
        tgt = answers
        return inp, tgt
                
                
    def data_iter(self, shuffle=True):
        if (shuffle == True):
            random.shuffle(self.data)
        seqlen = torch.zeros(self.data_num)
        for i in range(self.data_num):
            seqlen[i] = self.data[i].article.size(0)
        cache_start = 0
        while (cache_start < self.data_num):
            cache_end = min(cache_start + self.cache_size, self.data_num)
            cache_data = self.data[cache_start:cache_end]
            seql = seqlen[cache_start:cache_end]
            _, indices = torch.sort(seql, descending=True)
            batch_start = cache_start
            while (batch_start < cache_end):
                batch_end = min(batch_start + self.batch_size, cache_end)
                data_batch = indices[batch_start:batch_end]
                inp, tgt = self._batchify(cache_data, data_batch)
                inp = to_device(inp, self.device)
                tgt = to_device(tgt, self.device)
                yield inp, tgt
                batch_start += self.batch_size
            cache_start += self.cache_size
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bert cloth')
    args = parser.parse_args()
    '''
    data_collections = ['train', 'valid', 'test']
    for item in data_collections:    
        args.data_dir = './CLOTH/{}'.format(item)
        args.pre = args.post = 1
        args.save_name = './data/{}_3sents.pt'.format(item)
        args.bert_model = 'bert-base-uncased'
        data = Preprocessor(args)
    '''
    args.data_dir = './data/test_3sents.pt'
    args.bert_model = 'bert-base-uncased'
    args.cache_size = 512
    args.batch_size = 32
    train_data = Loader(args)
    for inp, tgt in train_data.data_iter():
        break
    #'''