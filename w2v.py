#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence
import time

################################## 1、写入文件，LineSentence 

tmp_query_id_dict ={}
fw = open('all_sentence.txt','w')
with open('/home/kesci/input/bytedance/first-round/train.csv') as f:
    for line in f:
        line = line.strip().split(',')
        query_id = line[0]
        if query_id not in tmp_query_id_dict:
            fw.write(line[1]+'\n')
        fw.write(line[3]+'\n')
        tmp_query_id_dict[query_id] = 1

tmp_query_id_dict ={}       
with open('/home/kesci/input/bytedance/first-round/test.csv') as f:
    for line in f:
        line = line.strip().split(',')
        query_id = line[0]
        if query_id not in tmp_query_id_dict:
            fw.write(line[1]+'\n')
        fw.write(line[3]+'\n')
        tmp_query_id_dict[query_id] = 1
fw.close()

print('train w2v')
sentences = LineSentence('all_sentence.txt')
model = Word2Vec(sentences, size=200, window=5, min_count=5)
model.wv.save_word2vec_format('w2v_model_50.txt', binary=False)


################################## 2、多个文件generator，LineSentence 
TRAIN_PATH = "/home/kesci/input/bytedance/train_final.csv"
TEST_PATH = "/home/kesci/input/bytedance/test_final_part1.csv"
CHUNKSIZE = 1000000
HEADER = ['qid', 'query', 'qtid', 'title', 'label']

start_time = time.time()
class CsvSentences:
    def __init__(self, fnames):
        self.fnames = fnames

    def __iter__(self):
        for fname in self.fnames:
            print(fname)
            with open(fname, 'r') as f:
                lines = f.readlines(256*1024*1024)
                while lines:
                    for line in lines:
                        line = line.strip().split(",")
                        query = line[1].strip().split()
                        title = line[3].strip().split()
                        yield query + ['[SEP]'] + title
                    lines = f.readlines(256*1024*1024)
            print('{}\'s reading completed'.format(fname.split('/')[-1]),", time used: ", timedelta(seconds=int(round(time.time() - start_time))))

temp_fnames = ["/home/kesci/input/bytedance/train_final.csv", 
               "/home/kesci/input/bytedance/test_final_part1.csv"]

if not os.path.exists('word2vec.model'):
    sentences = CsvSentences(temp_fnames)
    model = Word2Vec(sentences, size=200, window=5, min_count=5, workers=16)
    model.save('word2vec.model')
    model.wv.save_word2vec_format('word_embedding.txt', binary=False)