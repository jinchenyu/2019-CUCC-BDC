#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
from collections import Counter
import gc
import math
from multiprocessing import Pool
from multiprocessing import cpu_count

import sys
sys.append('../')
from util import *

def get_weight(count, eps=10000, min_count=2):
    return 0 if count < min_count else 1 / (count + eps)

loop = True
chunkSize = 30000000
counts = dict()
reader = pd.read_csv('/home/kesci/input/bytedance/train_final.csv', \
                    names=['query_id', 'query', 'query_title_id', 'title', 'label'], skiprows=900000011, iterator=True)
i = 0
while loop:
    print(i)
    i += 1
    try:
        chunk = reader.get_chunk(chunkSize)
        qs_texts = list(set(chunk['query'].tolist())) + chunk['title'].tolist()
        words = (" ".join(qs_texts)).split()
        tmp_counts = Counter(words)
        counts.update(tmp_counts)
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
print("chunk data done !")
print(i)


loop = True
reader = pd.read_csv('/home/kesci/input/bytedance/bytedance_contest.final_2.csv', names=['query_id', 'query', 'query_title_id', 'title'], iterator=True)
i = 0
while loop:
    print(i)
    i += 1
    try:
        chunk = reader.get_chunk(chunkSize)
        qs_texts = list(set(chunk['query'].tolist())) + chunk['title'].tolist()
        words = (" ".join(qs_texts)).split()
        tmp_counts = Counter(words)
        counts.update(tmp_counts)
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
print("chunk data done !")
print(i)
weights = {word: get_weight(count) for word, count in counts.items()}

print('dump done')
del counts
del tmp_counts
del words
del reader
del chunk
del qs_texts
gc.collect()

pickle.dump(weights, open('match_weights_1e_vs_1e.pkl','wb'))

print('load weights')
weights = pickle.load(open('match_weights_1e_vs_1e.pkl','rb'))



processor = 14
def feat_prepare():
    querys = pd.DataFrame(list(set(data['query_id'].values)), columns=['query_id'])
    l_data = len(querys)
    size = math.ceil(l_data / processor)  
    for i in range(processor):  
        start = size * i
        end = (i + 1) * size if (i + 1) * size < l_data else l_data
        query = querys[start:end]
        names['data_' + str(i)] = pd.merge(data, query, on='query_id').reset_index(drop=True)
        print(len(names['data_' + str(i)]))

def get_feat_by_multiprocessing(func):
    res = []
    p = Pool(processor)
    for i in range(processor):
        res.append(p.apply_async(func, args=( i,)))
    p.close()
    p.join()
    print('concat')
    data = pd.concat([i.get() for i in res])
    
    # 删除prepare的拆分数据
    for i in range(processor):
        del names['data_' + str(i)]
    gc.collect()
    
    return data


def word_shares(row):
    q1_list = str(row['query']).split()
    q2_list = str(row['title']).split()

    words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0]==i[1])/max(len(q1_list), len(q2_list))

    q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
    q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

    shared_2gram = q1_2gram.intersection(q2_2gram)

    if len(q1_2gram) + len(q2_2gram) == 0:
        R2gram = 0
    else:
        R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))

    return '{}:{}'.format(R2gram, words_hamming)

def word_match_share(q, t):
    if q == "PAD" or t == "PAD":
        return -1

    q_words = set(q.split())
    t_words = set(t.split())

    if len(q_words) == 0 or len(t_words) == 0:
        return 0.

    shared_words_in_q = [w for w in q_words if w in t_words]
    shared_words_in_t = [w for w in t_words if w in q_words]
    R = float(len(shared_words_in_q) + len(shared_words_in_t))/(len(q_words) + len(t_words))
    return R

def tfidf_word_match_share(q, t):
    if q == "PAD" or t == "PAD":
        return -1

    q_words = set(q.split())
    t_words = set(t.split())

    if len(q_words) == 0 or len(t_words) == 0:
        return 0.

    shared_weights = [weights.get(w, 0) for w in q_words if w in t_words] + [weights.get(w, 0) for w in t_words if w in q_words]
    total_weights = [weights.get(w, 0) for w in q_words] + [weights.get(w, 0) for w in t_words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def generate_feature(i):
    print(str(i) + ' processor started !')
    data = names['data_' + str(i)]

    with timer('word_shares'):
        data['word_shares'] = data.apply(word_shares, axis=1)
        data['shared_2gram'] = data['word_shares'].apply(lambda x: float(x.split(':')[0]))
        data['words_hamming'] = data['word_shares'].apply(lambda x: float(x.split(':')[1]))
        data = reduce_mem_usage(data)

    with timer('word_match_share'):
        data["word_match_share"] = data.apply(lambda x: word_match_share(x['query'], x['title']), axis=1)
        data["tfidf_word_match_share"] = data.apply(lambda x: tfidf_word_match_share(x['query'], x['title']), axis=1)
        data = reduce_mem_usage(data)

    return data


data = pd.read_csv('./data/data_10k2_lgb_base_10feat.csv', usecols=['query_id','query','title'],nrows=50000000)
print(data.shape)

names = locals()
feat_prepare()
data = get_feat_by_multiprocessing(generate_feature)

data = data.reset_index(drop=True)

drop_cols = ['word_shares']
base_cols = ['query_id','query','title']
data = data.drop(base_cols+drop_cols, axis=1)
print(data.shape)

data = reduce_mem_usage(data)
data.to_hdf('/home/kesci/work/features_lgb_10kw/train_only_match_features_part1.h5',key='df')

del data
gc.collect()


data = pd.read_csv('./data/data_10k2_lgb_base_10feat.csv', skiprows=50000000)
data.columns = ['query_id','query','query_title_id','title','label',
                'query_count','title_count','query_query_id_nunique',
                'is_query_has_2_query_id','same_count_query_title','len_query','len_title','is_query_in_title','query_in_title_query_ratio','query_in_title_title_ratio']
data = data[:-20000000]
data = data[['query_id','query','title']]
print(data.shape)

names = locals()
feat_prepare()
data = get_feat_by_multiprocessing(generate_feature)

data = data.reset_index(drop=True)

drop_cols = ['word_shares']
base_cols = ['query_id','query','title']
data = data.drop(base_cols+drop_cols, axis=1)
print(data.shape)

data = reduce_mem_usage(data)
data.to_hdf('/home/kesci/work/features_lgb_10kw/train_only_match_features_part2.h5',key='df')

del data
gc.collect()
print('train 1e done!')

