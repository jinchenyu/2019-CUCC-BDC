#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import numpy as np
import pandas as pd
from util import *


def load_train_data():
    data = pd.read_csv('./features_lgb_3kw/train_test_1etrain_basecount.csv')
    data = data[-29999994:]
    data = reduce_mem_usage(data)
    print(data.shape)
    data = data.reset_index(drop=True)

    y_train = data['label'].values
    y_train = np_utils.to_categorical(y_train)
    print(y_train.shape)

    base_feats = pd.read_csv('./features_lgb_3kw/train_only_1e_base_10feats.csv')
    base_feats = reduce_mem_usage(base_feats)
    print(base_feats.shape)
    base_feats = base_feats.reset_index(drop=True)

    match_features = pd.read_csv('./features_lgb_3kw/train_only_3k2_new_match_features.csv')
    match_features = reduce_mem_usage(match_features)
    print(match_features.shape)
    match_features = match_features.reset_index(drop=True)
    len_features = pd.read_csv('./features_lgb_3kw/train_test_3k2_len_features.csv',nrows=29999994)
    len_features = reduce_mem_usage(len_features)
    print(len_features.shape)
    len_features = len_features.reset_index(drop=True)
    emb_features = pd.read_csv('./features_lgb_3kw/train_test_3k2_new_embedding_simi_features_v2.csv',nrows=29999994) 
    emb_features = reduce_mem_usage(emb_features)
    print(emb_features.shape)
    emb_features = emb_features.reset_index(drop=True)
    simifeat_features = pd.read_csv('./features_lgb_3kw/train_test_3k2_simifeats_features.csv',nrows=29999994)
    simifeat_features = reduce_mem_usage(simifeat_features)
    print(simifeat_features.shape)
    simifeat_features = simifeat_features.reset_index(drop=True)
    fuzzy_features = pd.read_csv('./features_lgb_3kw/train_test_3k2_fuzzydifflib_features.csv',nrows=29999994)
    fuzzy_features = reduce_mem_usage(fuzzy_features)
    print(fuzzy_features.shape)
    fuzzy_features = fuzzy_features.reset_index(drop=True)

    data = pd.concat([data,
                      base_feats,
                      match_features,
                      len_features,
                      emb_features,
                      simifeat_features,
                      fuzzy_features],
                      axis=1
                      )
    print(data.shape)
    data = reduce_mem_usage(data)

    del base_feats
    del match_features
    del len_features
    del emb_features
    del simifeat_features
    del fuzzy_features
    gc.collect()

    # x1 = [[int(i) for i in v.strip().split(' ') if i in model] for v in data['query'].values]
    # x1 = pad_sequences(x1, maxlen=5)
    # x2 = [[int(i) for i in v.strip().split(' ') if i in model] for v in data['title'].values]
    # x2 = pad_sequences(x2, maxlen=15)
    x1 = np.load('./features_lgb_3kw/3kw_x1_5.npy')
    x2 = np.load('./features_lgb_3kw/3kw_x2_15.npy')

    no_use = ['query_id', 'query', 'title', 'query_query_id_nunique', 'query_id_query_title_id_max'] + ['label']
    usefeats = [f for f in data.columns if f not in no_use]
    print(len(usefeats))

    train_feats = data[usefeats]
    del data
    gc.collect()
    train_feats = train_feats.fillna(0)

    return x1, x2, train_feats


def get_1e_test_part(mode):
    data = pd.read_csv('./features_final_test/test_final_1e_base_count.csv')
    if mode == 1:
        data = data[:50000000]
    else:
        data = data[50000000:]
    print(data.shape)
    # print(data.columns)
    data = reduce_mem_usage(data)
    data = data.reset_index(drop=True)
    
    base_feats = pd.read_csv('./features_final_test/test_final_1e_base_10feats.csv')
    if mode == 1:
        base_feats = base_feats[:50000000]
    else:
        base_feats = base_feats[50000000:]
    print(base_feats.shape)
    # print(base_feats.columns)
    base_feats = reduce_mem_usage(base_feats)
    base_feats = base_feats.reset_index(drop=True)
    
    match_features = pd.read_csv('./features_final_test/test_final_10kw_match_features_part{}.csv'.format(mode))
    print(match_features.shape)
    match_features = reduce_mem_usage(match_features)
    match_features = match_features.reset_index(drop=True)
    len_features = pd.read_csv('./features_final_test/test_final_10kw_len_features_part{}.csv'.format(mode))
    print(len_features.shape)
    len_features = reduce_mem_usage(len_features)
    len_features = len_features.reset_index(drop=True)
    emb_features = pd.read_csv('./features_final_test/test_final_10kw_embedding_emb_features_part{}.csv'.format(mode)) 
    print(emb_features.shape)
    emb_features = reduce_mem_usage(emb_features)
    emb_features = emb_features.reset_index(drop=True)
    simifeat_features = pd.read_csv('./features_final_test/test_final_10kw_simifeats_features_part{}.csv'.format(mode))
    print(simifeat_features.shape)
    simifeat_features = reduce_mem_usage(simifeat_features)
    simifeat_features = simifeat_features.reset_index(drop=True)
    fuzzy_features = pd.read_csv('./features_final_test/test_final_10kw_fuzzydifflib_features_part{}.csv'.format(mode))
    print(fuzzy_features.shape)
    fuzzy_features = reduce_mem_usage(fuzzy_features)
    fuzzy_features = fuzzy_features.reset_index(drop=True)
    
    data = pd.concat([data,
                      base_feats,
                      match_features,
                      len_features,
                      emb_features,
                      simifeat_features,
                      fuzzy_features],
                      axis=1
                      )
    print(data.shape)
    data = reduce_mem_usage(data)
    
    del base_feats
    del match_features
    del len_features
    del emb_features
    del simifeat_features
    del fuzzy_features
    gc.collect()

    # x1 = [[int(i) for i in v.strip().split(' ') if i in model] for v in data['query'].values]
    # x1 = pad_sequences(x1, maxlen=5)
    # x2 = [[int(i) for i in v.strip().split(' ') if i in model] for v in data['title'].values]
    # x2 = pad_sequences(x2, maxlen=15)

    x1 = np.load('./features_final_test/test_part{}_x1.npy'.format(mode))
    x2 = np.load('./features_final_test/test_part{}_x2.npy'.format(mode))
    print(x1.shape, x2.shape)

    
    no_use = ['query_id', 'query', 'title', 'query_query_id_nunique', 'query_id_query_title_id_max'] + ['label']
    usefeats = [f for f in data.columns if f not in no_use]
    print(len(usefeats))
    
    train_feats = data[usefeats]
    del data
    gc.collect()
    train_feats = train_feats.fillna(0)

    return x1, x2, train_feats