#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import gc
import jieba, os, Levenshtein, time
from scipy import sparse
import math
from util import *
from multiprocessing import Pool
from multiprocessing import cpu_count


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
    data = pd.concat([i.get() for i in res])
    
    # 删除prepare的拆分数据
    for i in range(processor):
        del names['data_' + str(i)]
    gc.collect()
    
    return data

def generate_feature(i):
    print(str(i) + ' processor started !')
    data = names['data_' + str(i)]
    similarity_func = [Levenshtein.ratio, Levenshtein.distance, lcsubstr_lens, lcseque_lens]
    with timer('similarity_func'):
        for func in similarity_func:
            print(func.__name__)
            data[func.__name__ + '_similarity_' + 'query' + '_with_' + 'title'] = data.apply(lambda row: func(row['query'], row['title']), axis=1)
            data = reduce_mem_usage(data)

    return data
    
    
data = pd.read_csv('/home/kesci/input/bytedance/bytedance_contest.final_2.csv', nrows=50000000, names=['query_id', 'query', 'query_title_id', 'title'])
print(data.head())
print(data.shape)

names = locals()
feat_prepare()
data = get_feat_by_multiprocessing(generate_feature)
data = data.reset_index(drop=True)

drop_cols = []
base_cols = ['query_id','query','title','query_title_id']
data = data.drop(base_cols+drop_cols, axis=1)
print(data.shape)

data = reduce_mem_usage(data)
data.to_csv('/home/kesci/work/features_final_test/test_final_10kw_simifeats_features_part1.csv', index=False)


data = pd.read_csv('/home/kesci/input/bytedance/bytedance_contest.final_2.csv', skiprows=50000000, names=['query_id', 'query', 'query_title_id', 'title'])
print(data.head())
print(data.shape)

names = locals()
feat_prepare()
data = get_feat_by_multiprocessing(generate_feature)
data = data.reset_index(drop=True)

drop_cols = []
base_cols = ['query_id','query','title','query_title_id']
data = data.drop(base_cols+drop_cols, axis=1)
print(data.shape)

data = reduce_mem_usage(data)
data.to_csv('/home/kesci/work/features_final_test/test_final_10kw_simifeats_features_part2.csv', index=False)


