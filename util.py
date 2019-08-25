#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import time
from contextlib import contextmanager

@contextmanager # 上下文管理器
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.6f}s".format(title, time.time() - t0))

def reduce_mem_usage(df, verbose=True):
    """
    降低num类特征内存
    :param df:
    :param verbose:
    :return:
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def feat_nunique(df, df_feature, fe, value, name=""):
    df_count = df_feature.groupby(fe)[value].nunique().reset_index(name=name)    # 特征fe，在value中次数统计
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_count(df, df_feature, fe, value, name=""):
    df_count = df_feature.groupby(fe)[value].count().reset_index(name=name)    # 特征fe，在value中次数统计
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_sum(df, df_feature, fe, value, name=""):
    df_count = df_feature.groupby(fe)[value].sum().reset_index(name=name)    # 特征fe，在value中次数统计
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_mean(df, df_feature, fe, value, name=""):
    df_count = df_feature.groupby(fe)[value].mean().reset_index(name=name)    # 特征fe，在value中次数统计
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_median(df, df_feature, fe, value, name=""):
    df_count = df_feature.groupby(fe)[value].median().reset_index(name=name)    # 特征fe，在value中次数统计
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_var(df, df_feature, fe, value, name=""):
    df_count = df_feature.groupby(fe)[value].var().reset_index(name=name)    # 特征fe，在value中次数统计
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_min(df, df_feature, fe, value, name=""):
    df_count = df_feature.groupby(fe)[value].min().reset_index(name=name)    # 特征fe，在value中次数统计
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_max(df, df_feature, fe, value, name=""):
    df_count = df_feature.groupby(fe)[value].max().reset_index(name=name)    # 特征fe，在value中次数统计
    df = df.merge(df_count, on=fe, how="left")
    return df

def lcsubstr_lens(s1, s2): 
    s1 = s1.split()
    s2 = s2.split()
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  #生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax=0   #最长匹配的长度
    p=0  #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
                if m[i+1][j+1]>mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return mmax

def lcseque_lens(s1, s2): 
    s1 = s1.split()
    s2 = s2.split()
     # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [ [ 0 for x in range(len(s2)+1) ] for y in range(len(s1)+1) ] 
    # d用来记录转移方向
    d = [ [ None for x in range(len(s2)+1) ] for y in range(len(s1)+1) ] 

    for p1 in range(len(s1)): 
        for p2 in range(len(s2)): 
            if s1[p1] == s2[p2]:            #字符匹配成功，则该位置的值为左上方的值加1
                m[p1+1][p2+1] = m[p1][p2]+1
                d[p1+1][p2+1] = 'ok'          
            elif m[p1+1][p2] > m[p1][p2+1]:  #左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1+1][p2+1] = m[p1+1][p2] 
                d[p1+1][p2+1] = 'left'          
            else:                           #上值大于左值，则该位置的值为上值，并标记方向up
                m[p1+1][p2+1] = m[p1][p2+1]   
                d[p1+1][p2+1] = 'up'         
    (p1, p2) = (len(s1), len(s2)) 
    s = [] 
    while m[p1][p2]:    #不为None时
        c = d[p1][p2]
        if c == 'ok':   #匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1-1])
            p1 -= 1
            p2 -= 1 
        if c == 'left':  #根据标记，向左找下一个
            p2 -= 1
        if c == 'up':   #根据标记，向上找下一个
            p1 -= 1
    return len(s)

def compute_convert(pos, sums, label):
    if np.isnan(sums): # not oppear in train
        return -1
    if label != -1 and sums == 1: # only oppear once
        return -1
    if label == 1:
        return (pos - 1) / (sums - 1)
    elif label == 0:
        return pos / (sums - 1)
    else:
        return pos / sums

def find_longest_prefix(str_list):
    if not str_list:
        return ''
    str_list.sort(key = lambda x: len(x))
    shortest_str = str_list[0]
    max_prefix = len(shortest_str)
    flag = 0
    for i in range(max_prefix):
        for one_str in str_list:
            if one_str[i] != shortest_str[i]:
                return shortest_str[:i]
                break
    return shortest_str


np.random.seed(0)
class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(tries, success)
        #print 'mean and variance: ', mean, var
        #self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''moment estimation'''
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i])/(tries[i] + 0.000000001))
        mean = sum(ctr_list)/len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr-mean, 2)

        return mean, var/(len(ctr_list)-1)