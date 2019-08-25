#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
lgb baseline feature
10特征，初赛线上0.5813，复赛线上0.5835
"""

import pandas as pd
import numpy as np
import gc
from multiprocessing import Pool
import math
from util import *


TARGET = 'label'
processor = 16

# 前1e数据，根据query id划分
train = pd.read_csv('/home/kesci/input/bytedance/train_final.csv', \
                    names=['query_id', 'query', 'query_title_id', 'title', 'label'], skiprows=900000011)
test = pd.read_csv('/home/kesci/input/bytedance/bytedance_contest.final_2.csv', \
                    names=['query_id', 'query', 'query_title_id', 'title'])
print(train.shape)
print(test.shape)

test[TARGET] = -1
data = pd.concat([train, test]).reset_index(drop=True)

del train, test
gc.collect()

data = reduce_mem_usage(data)

data = feat_count(data, data, ['query'], 'query_id', 'query_count')
data = feat_count(data, data, ['title'], 'query_id', 'title_count')

data = feat_nunique(data, data, ['query'], 'query_id', 'query_query_id_nunique')
data['is_query_has_2_query_id'] = data['query_query_id_nunique'].apply(lambda x:0 if x == 1 else 1)

# train, test分两部分保存
data[data['label']!=-1].to_csv('/home/kesci/work/features_lgb_3kw/train_1e_base10_count.csv',index=False)
data[data['label']==-1].to_csv('/home/kesci/work/features_final_test/test_final_1e_base10_count.csv',index=False)

def feat(i):
    print(str(i) + ' processor started !')
    data = names['data_' + str(i)]
    
    with timer('same_count_query_title'):
        def f(x):
            cnt = 0
            for i in x.query.split():
                if i in x.title.split():
                    cnt += 1
            return cnt
        data['same_count_query_title'] = data.apply(f, axis=1)
        
    with timer('split'): 
        data['len_query'] = data['query'].apply(lambda x: len(x.split()))
        data['len_title'] = data['title'].apply(lambda x: len(x.split()))
    
    with timer('is_query_in_title'):
        data['is_query_in_title'] = data.apply(lambda x: 1 if x.query in x.title else 0, axis=1)
        
    with timer('ratio'):
        data['query_in_title_query_ratio'] = data['same_count_query_title'] / data['len_query']
        data['query_in_title_title_ratio'] = data['same_count_query_title'] / data['len_title']
    
    return data
    
def get_feat():
    res = []
    p = Pool(processor) 
    for i in range(processor):
        res.append(p.apply_async(feat, args=( i,)))
    p.close()
    p.join()
    data = pd.concat([i.get() for i in res])
    
    for i in range(processor):
        del names['data_' + str(i)]
    gc.collect()
    
    return data
    
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


names = locals()
feat_prepare()
data = get_feat()
data = reduce_mem_usage(data)

no_use = ['query_id', 'query', 'query_title_id', 'title']
feats = [f for f in data.columns if f not in no_use]
print(len(feats))

data[data['label']!=-1][feats].to_csv('/home/kesci/work/features/train_1e_base10.csv',index=False)
data[data['label']==-1][feats].to_csv('/home/kesci/work/features/test_final_1e_base10.csv',index=False)
