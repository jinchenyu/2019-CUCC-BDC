#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import gc
import math
from multiprocessing import Pool
from multiprocessing import cpu_count

import sys
sys.append('../')
from util import *


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

#统计query在title中的位置
def get_query_start_position(q, t):
    query_0 = q.split()[0]
    return str(t).strip().find(str(query_0))
def get_query_end_position(q, t):
    query_0 = q.split()[-1]
    return str(t).strip().find(str(query_0))

#统计query在title中的位置
def get_query_start_position_split(q, t):
    query_0 = q.split()[0]
    if query_0 in t.strip().split():
        return t.strip().split().index(query_0)
    else:
        return -1
def get_query_end_position_split(q, t):
    query_0 = q.split()[-1]
    if query_0 in t.strip().split():
        return t.strip().split().index(query_0)
    else:
        return -1

#统计query在title中的位置
def get_query_real_start_position(q, t):
    query_0 = 'FLAG'
    for i in q.split():
        if i not in t:
            continue
        query_0 = i
        break
    return str(t).strip().find(str(query_0))
def get_query_real_end_position(q, t):
    query_0 = 'FLAG'
    for i in q.split()[::-1]:
        if i not in t:
            continue
        query_0 = i
        break
    return str(t).strip().find(str(query_0))

#统计query在title中的位置
def get_query_real_start_position_split(q, t):
    query_0 = 'FLAG'
    for i in q.split():
        if i not in t:
            continue
        query_0 = i
        break
    try:
        return t.strip().split().index(query_0)
    except:
        return -1
def get_query_real_end_position_split(q, t):
    query_0 = 'FLAG'
    for i in q.split()[::-1]:
        if i not in t:
            continue
        query_0 = i
        break
    try:
        return t.strip().split().index(query_0)
    except:
        return -1

def generate_feature(i):
    print(str(i) + ' processor started !')
    data = names['data_' + str(i)]

    with timer('query_title_len'):
        data['word_len1'] = data['query'].map(lambda x: len(str(x).split()))
        data['word_len2'] = data['title'].map(lambda x: len(str(x).split()))
        data['word_str_len_1'] = data['query'].map(lambda x: len(str(x)))
        data['word_str_len_2'] = data['title'].map(lambda x: len(str(x)))
        data['word_len2_diff_word_len1'] = data['word_len2'] - data['word_len1']
        data['word_len2_diff_word_len1_str'] = data['word_str_len_2'] - data['word_str_len_1']
        data['word_len1_word_len2_ratio'] = data['word_len1'] / data['word_len2']
        data['word_len1_word_len2_ratio_str'] = data['word_str_len_2'] / data['word_str_len_1']
        data['word_len_diff_ratio'] = data.apply(lambda row: abs(row.word_len1-row.word_len2)/(row.word_len1+row.word_len2), axis=1)
        data = reduce_mem_usage(data)

    with timer('query_title_position'):
        data["get_query_start_position"] = data.apply(lambda x: get_query_start_position(x['query'], x['title']), axis=1)
        data["get_query_end_position"] = data.apply(lambda x: get_query_end_position(x['query'], x['title']), axis=1)
        data["get_query_position_diff"] = data['get_query_end_position'] - data['get_query_start_position']
        data["get_query_position_diff_ratio"] = data['get_query_position_diff'] / data['word_str_len_1']
        data["get_query_start_position_ratio"] = data['get_query_start_position'] / data['word_str_len_2']
        data["get_query_end_position_ratio"] = data['get_query_end_position'] / data['word_str_len_2']
        data = reduce_mem_usage(data)

    with timer('query_title_real_position'):
        data["get_query_start_position_real"] = data.apply(lambda x: get_query_real_start_position(x['query'], x['title']), axis=1)
        data["get_query_end_position_real"] = data.apply(lambda x: get_query_real_end_position(x['query'], x['title']), axis=1)
        data["get_query_position_diff_real"] = data['get_query_end_position_real'] - data['get_query_start_position_real']
        data["get_query_position_diff_ratio_real"] = data['get_query_position_diff_real'] / data['word_str_len_1']
        data["get_query_start_position_ratio_real"] = data['get_query_start_position_real'] / data['word_str_len_2']
        data["get_query_end_position_ratio_real"] = data['get_query_end_position_real'] / data['word_str_len_2']
        data = reduce_mem_usage(data)

    with timer('query_title_position_split'):
        data["get_query_start_position_split"] = data.apply(lambda x: get_query_start_position_split(x['query'], x['title']), axis=1)
        data["get_query_end_position_split"] = data.apply(lambda x: get_query_end_position_split(x['query'], x['title']), axis=1)
        data["get_query_position_diff_split"] = data['get_query_end_position_split'] - data['get_query_start_position_split']
        data["get_query_position_diff_ratio_split"] = data['get_query_position_diff_split'] / data['word_len1']
        data["get_query_start_position_ratio_split"] = data['get_query_start_position_split'] / data['word_len2']
        data["get_query_end_position_ratio_split"] = data['get_query_end_position_split'] / data['word_len2']
        data = reduce_mem_usage(data)

    with timer('query_title_real_position_split'):
        data["get_query_start_position_real_split"] = data.apply(lambda x: get_query_real_start_position_split(x['query'], x['title']), axis=1)
        data["get_query_end_position_real_split"] = data.apply(lambda x: get_query_real_end_position_split(x['query'], x['title']), axis=1)
        data["get_query_position_diff_real_split"] = data['get_query_end_position_real_split'] - data['get_query_start_position_real_split']
        data["get_query_position_diff_ratio_real_split"] = data['get_query_position_diff_real_split'] / data['word_len1']
        data["get_query_start_position_ratio_real_split"] = data['get_query_start_position_real_split'] / data['word_len2']
        data["get_query_end_position_ratio_real_split"] = data['get_query_end_position_real_split'] / data['word_len2']
        data = reduce_mem_usage(data)

    with timer('feat row'):
        data["get_query_position_diff_ratio_row"] = data['get_query_position_diff'] / data['word_len1']
        data["get_query_start_position_ratio_row"] = data['get_query_start_position'] / data['word_len2']
        data["get_query_end_position_ratio_row"] = data['get_query_end_position'] / data['word_len2']
        data = reduce_mem_usage(data)

    return data


data = pd.read_csv('/home/kesci/input/bytedance/bytedance_contest.final_2.csv', names=['query_id', 'query', 'query_title_id', 'title'], nrows=50000000)
print(data.shape)

names = locals()
feat_prepare()
data = get_feat_by_multiprocessing(generate_feature)
data = data.reset_index(drop=True)

drop_cols = ['word_len1','word_len2']
base_cols = ['query_id','query','title','query_title_id']
data = data.drop(base_cols+drop_cols, axis=1)
print(data.shape)

data = reduce_mem_usage(data)
data.to_csv('/home/kesci/work/features_final_test/test_final_10kw_len_features_part1.csv', index=False)

del data
gc.collect()



data = pd.read_csv('/home/kesci/input/bytedance/bytedance_contest.final_2.csv', skiprows=50000000, names=['query_id', 'query', 'query_title_id', 'title'])
print(data.head())
print(data.shape)

names = locals()
feat_prepare()
data = get_feat_by_multiprocessing(generate_feature)
data = data.reset_index(drop=True)

drop_cols = ['word_len1','word_len2']
base_cols = ['query_id','query','title','query_title_id']
data = data.drop(base_cols+drop_cols, axis=1)
print(data.shape)

data = reduce_mem_usage(data)
data.to_csv('/home/kesci/work/features_final_test/test_final_10kw_len_features_part2.csv', index=False)

