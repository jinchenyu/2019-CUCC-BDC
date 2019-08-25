#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import gc
import math
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import lightgbm as lgb
import datetime

from util import *

TARGET = 'label'
processor = 16 # 使用线程数

# quac计算，多线程
def comput_query_auc(i, data):
    # print(str(i) + ' processor started !')
    score = []
    for index, group in data.groupby(['query_id']):
        try:
            score.append(roc_auc_score(y_true=group['label'],y_score=group['pred']))
        except ValueError:
            score.append(0.5)
    
    # print(str(i) + ' processor finished !')
    return score

def get_qauc(df):
    # 输入df包含query_id, label, pred

    names = locals()
    
    querys = pd.DataFrame(list(set(df['query_id'].values)), columns=['query_id'])
    l_data = len(querys) 
    size = math.ceil(l_data / processor)
    for i in range(processor): 
        start = size * i
        end = (i + 1) * size if (i + 1) * size < l_data else l_data
        query = querys[start:end]
        t_data = pd.merge(df, query, on='query_id').reset_index(drop=True)
        names['qauc_'+str(i)] = t_data
        # print(len(query))
    
    res = []
    p = Pool(processor)
    for i in range(processor):
        res.append(p.apply_async(comput_query_auc, args=(i, names['qauc_'+str(i)])))
    p.close()
    p.join()
    
    score = []
    for i in res:
        score += i.get()
        
    qauc = np.mean(score)
    print('qauc ', qauc)
    return qauc


TRAIN_PATH = './features_lgb_10kw/train_data_1e.csv'
VALID_PATH = './features_lgb_10kw/valid_data_3m.csv'

# 构造data type
dtype = pd.read_csv('./dtypes.csv')
DATA_TYPE = dict()
for index, row in dtype.iterrows():
    DATA_TYPE[row['col_name']] = row['dtype']
del dtype
gc.collect()


train = pd.read_csv(TRAIN_PATH, dtype=DATA_TYPE, nrows=TRAIN_SIZE)
test = pd.read_csv(TEST_PATH, dtype=DATA_TYPE)
data = pd.concat([train, test])


def display_importances(feature_importance_df_, score):
    ft = feature_importance_df_[["feature", "split", "gain"]].groupby("feature").mean().sort_values(by="gain",
                                                                                                    ascending=False)
    print(ft)
    # import os
    # if not os.path.exists('/feat_importance'):
    #     print('./feat_importance')
    #     os.makedirs('./feat_importance')
    now = datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    ft.to_csv('./feat_importance/importance_lightgbm_{}_{}.csv'.format(score, now), index=True)
    # cols = ft[:40].index
    # best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

def kfold_lightgbm_split_query_id(params, df, predictors, target, num_folds, stratified=False,
                   objective='', metrics='', debug=False,
                   feval=None, early_stopping_rounds=100, num_boost_round=10000, verbose_eval=1,
                   categorical_features=None):
    
    lgb_params = params

    train_df = df[df[target] != -1]
    test_df = df[df[target] == -1]

    del df
    gc.collect()

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df[predictors].shape,
                                                                      test_df[predictors].shape))

    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1)

    oof_preds = np.zeros((train_df.shape[0]))
    sub_preds = np.zeros((test_df.shape[0]))
    feature_importance_df = pd.DataFrame()
    feats = predictors
    cv_resul = []

    print('train query id nunique:', train_df['query_id'].nunique())
    train_querys = train_df[['query_id']].drop_duplicates()

    # 根据query id划分
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_querys, [0]*train_querys.shape[0])):
        if (USE_KFOLD == False) and (n_fold == 1):
            break
        
        train_x, train_y = train_df[train_df['query_id'].isin(list(train_querys.iloc[train_idx]['query_id']))][feats], \
                          train_df[train_df['query_id'].isin(list(train_querys.iloc[train_idx]['query_id']))][target]
        valid_x, valid_y = train_df[train_df['query_id'].isin(list(train_querys.iloc[valid_idx]['query_id']))][feats], \
                          train_df[train_df['query_id'].isin(list(train_querys.iloc[valid_idx]['query_id']))][target]
        
        print(train_x.shape, valid_x.shape)

        train_y_t = train_y.values
        valid_y_t = valid_y.values

        xgtrain = lgb.Dataset(train_x.values, label=train_y_t,
                                      feature_name=predictors,
                                      categorical_feature=categorical_features
                                      )
        del train_x, train_y_t
        gc.collect()
        
        xgvalid = lgb.Dataset(valid_x.values, label=valid_y_t,
                                      feature_name=predictors,
                                      categorical_feature=categorical_features
                                      )
        del valid_y_t
        gc.collect()

        clf = lgb.train(lgb_params,
                            xgtrain,
                            valid_sets=[xgtrain, xgvalid],
                            valid_names=['train', 'valid'],
                            num_boost_round=num_boost_round,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=verbose_eval,
                            )
        
        val_preds = clf.predict(valid_x, num_iteration=clf.best_iteration)
        oof_preds[valid_x.index] = val_preds
        
        # qauc指标计算
        qauc_val_df = train_df[train_df['query_id'].isin(list(train_querys.iloc[valid_idx]['query_id']))][['query_id', 'label']]
        qauc_val_df['pred'] = val_preds
        with timer('get qauc'):
            get_qauc(qauc_val_df)
        
        if USE_KFOLD == False:
            sub_preds = clf.predict(test_df[feats], num_iteration=clf.best_iteration)
        else:
            sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration) / folds.n_splits

        gain = clf.feature_importance('gain')
        fold_importance_df = pd.DataFrame({'feature': clf.feature_name(),
                                                            'split': clf.feature_importance('split'),
                                                            'gain': 100 * gain / gain.sum(),
                                                            'fold': n_fold,
                                                           }).sort_values('gain', ascending=False)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        result = clf.best_score['valid']['binary_logloss']
        print('Fold %2d binary_logloss : %.6f' % (n_fold + 1, result))
        cv_resul.append(round(result, 5))
        gc.collect()

    with open('result.txt', 'a') as f:
        f.write(target + ' cv:' + str(cv_resul) + ' mean:' + str(np.mean(cv_resul)) + '\n')
    print(target + ' cv:' + str(cv_resul) + ' mean:' + str(np.mean(cv_resul)) + '\n')
    
    score = np.array(cv_resul).mean()
    display_importances(feature_importance_df, score)

    return sub_preds



no_use = ['query_id', 'query', 'title', 'query_query_id_nunique', ] + [TARGET]
cate_feat = []

feats = [f for f in train.columns if f not in no_use]
categorical_columns = [f for f in cate_feat if f not in no_use]

print(cate_feat)
print(len(cate_feat))
print(len(feats))

params = {
    "learning_rate": 0.1,
    'boosting': 'gbdt',
    "objective": "binary",
    'seed': 1,
    
    # 'device':'gpu',
    # 'gpu_platform_id':0,
    # 'gpu_device_id':0
    
    'n_jobs':16,
}

USE_KFOLD = True
preds = kfold_lightgbm_split_query_id_only_train(params, train, feats, TARGET, 5, USE_KFOLD, categorical_features=categorical_columns, 
                                                                    early_stopping_rounds=30, num_boost_round=120)


submission = pd.read_csv('/home/kesci/input/bytedance/bytedance_contest.final_2.csv', \
                    names=['query_id', 'query', 'query_title_id', 'title'],
                    usecols=['query_id','query_title_id'])
submission['pred'] = preds
submission.to_csv('./submission/lgb_submission.csv',index=False,header=None)





