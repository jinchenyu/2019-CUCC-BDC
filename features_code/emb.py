#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import gensim
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import gc
import math
import gensim
from multiprocessing import Pool
from multiprocessing import cpu_count

import sys
sys.append('../')
from util import *

model = gensim.models.Word2Vec.load('/home/kesci/work/hmz/example_v2/word2vec_all_v2.model')


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
    
    for i in range(processor):
        del names['data_' + str(i)]
    gc.collect()
    
    return data


def wmd(s1, s2):
    s1 = str(s1).strip().split()
    s2 = str(s2).strip().split()
    s1 = [w for w in s1 if w in model]
    s2 = [w for w in s2 if w in model]
    return model.wmdistance(s1, s2)

def sent2vec(s):
    words = str(s).strip().split()
    words = [w for w in words if w in model]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

def get_w2v_simi(query, title):
    q_vec = np.nan_to_num(sent2vec(query))
    t_vec = np.nan_to_num(sent2vec(title))

    w2v_consine = cosine(q_vec, t_vec)
    w2v_cityblock = cityblock(q_vec, t_vec)
    w2v_jaccard = jaccard(q_vec, t_vec)
    w2v_canberra = canberra(q_vec, t_vec)
    w2v_euclidean = euclidean(q_vec, t_vec)
    w2v_minkowski = minkowski(q_vec, t_vec)
    w2v_braycurtis = braycurtis(q_vec, t_vec)

    w2v_skew_qvec = skew(q_vec)
    w2v_skew_tvec = skew(t_vec)
    w2v_kur_qvec = kurtosis(q_vec)
    w2v_kur_tvec = kurtosis(t_vec)

    outlist = [w2v_consine,
               w2v_cityblock,
               w2v_jaccard,
               w2v_canberra,
               w2v_euclidean,
               w2v_minkowski,
               w2v_braycurtis,
               w2v_skew_qvec,
               w2v_skew_tvec,
               w2v_kur_qvec,
               w2v_kur_tvec
               ]
    outformat = ':'.join(['{}']*len(outlist))

    return outformat.format(*outlist)

def generate_feature(i):
    print(str(i) + ' processor started !')
    data = names['data_' + str(i)]

    with timer('w2v simi'):
        data['w2v_model_simi'] = data.apply(lambda row: get_w2v_simi(row['query'], row['title']), axis=1)
        data['w2v_consine'] = data['w2v_model_simi'].apply(lambda x: float(x.split(':')[0]))
        data['w2v_cityblock'] = data['w2v_model_simi'].apply(lambda x: float(x.split(':')[1]))
        data['w2v_jaccard'] = data['w2v_model_simi'].apply(lambda x: float(x.split(':')[2]))
        data['w2v_canberra'] = data['w2v_model_simi'].apply(lambda x: float(x.split(':')[3]))
        data['w2v_euclidean'] = data['w2v_model_simi'].apply(lambda x: float(x.split(':')[4]))
        data['w2v_minkowski'] = data['w2v_model_simi'].apply(lambda x: float(x.split(':')[5]))
        data['w2v_braycurtis'] = data['w2v_model_simi'].apply(lambda x: float(x.split(':')[6]))
        data['w2v_skew_qvec'] = data['w2v_model_simi'].apply(lambda x: float(x.split(':')[7]))
        data['w2v_skew_tvec'] = data['w2v_model_simi'].apply(lambda x: float(x.split(':')[8]))
        data['w2v_kur_qvec'] = data['w2v_model_simi'].apply(lambda x: float(x.split(':')[9]))
        data['w2v_kur_tvec'] = data['w2v_model_simi'].apply(lambda x: float(x.split(':')[10]))
        data = reduce_mem_usage(data)

    return data



# 每隔5kw做一次特征，不然内存不够
data = pd.read_csv('/home/kesci/input/bytedance/bytedance_contest.final_2.csv', names=['query_id', 'query', 'query_title_id', 'title'], nrows=50000000)
print(data.shape)

names = locals()
feat_prepare()
data = get_feat_by_multiprocessing(generate_feature)
data = data.reset_index(drop=True)

drop_cols = ['w2v_model_simi']
base_cols = ['query_id','query','title','query_title_id']
data = data.drop(base_cols+drop_cols, axis=1)
print(data.shape)

data = reduce_mem_usage(data)
data.to_hdf('/home/kesci/work/features_final_test/test_final_10kw_embedding_emb_features_part1.h5',key='df')

del data
gc.collect()


data = pd.read_csv('/home/kesci/input/bytedance/bytedance_contest.final_2.csv', skiprows=50000000, names=['query_id', 'query', 'query_title_id', 'title'])
print(data.head())
print(data.shape)

names = locals()
feat_prepare()
data = get_feat_by_multiprocessing(generate_feature)
data = data.reset_index(drop=True)

drop_cols = ['w2v_model_simi']
base_cols = ['query_id','query','title','query_title_id']
data = data.drop(base_cols+drop_cols, axis=1)
print(data.shape)

data = reduce_mem_usage(data)
data.to_hdf('/home/kesci/work/features_final_test/test_final_10kw_embedding_emb_features_part2.h5',key='df')







