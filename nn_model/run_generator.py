#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import gc
import gensim
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from util import *
from model_store import *
from load_data import *

import random
random.seed(802)
np.random.seed(802)
tf.set_random_seed(802)


######################## 1、embedding_matrix
model = gensim.models.Word2Vec.load('/home/kesci/work/hmz/example_v2/word2vec_all_v2.model')

max_features = 1936937

embedding_matrix = np.zeros((max_features, 200))
for word in model.wv.vocab.keys():
    try:
        embedding_matrix[int(word)] = model[word]
    except:
        continue

sequence_length = 20
max_len_1 = 5
max_len_2 = 15
features_len = 71

######################## 2、import model
model_nn = m_bilstm_attention_esim_add_feature_model(max_features, 
                                                       max_len_1, 
                                                       max_len_2,
                                                       features_len,
                                                       embedding_matrix
                                                       )


TRAIN_PATH = './features_lgb_10kw/train_data_1e.csv'
TRAIN_SIZE = 96999989
VALID_PATH = './features_lgb_10kw/valid_data_3m.csv'
VALID_SIZE = 3000000
# TEST_PATH = '/home/kesci/work/features_lgb_10kw/all_feats_10kw_test.csv'
# TEST_SIZE = 20000000
BATCH_SIZE =  8192


def my_generator(file_path, batch_size):

    while True:
        
        un_training_data = pd.read_csv(file_path, nrows=TRAIN_SIZE, chunksize=batch_size)
        
        for index_in_chunk, chunk in enumerate(un_training_data):
            # taking the values from panda into numpy array
            x1 = [[int(i) for i in v.strip().split(' ') if i in model] for v in chunk['query'].values]
            x1 = pad_sequences(x1, maxlen=5)
            x2 = [[int(i) for i in v.strip().split(' ') if i in model] for v in chunk['title'].values]
            x2 = pad_sequences(x2, maxlen=15)

            y_train = chunk['label'].values
            y_train = np_utils.to_categorical(y_train)
            
            no_use = ['query_id', 'query', 'title', 'query_query_id_nunique', 'query_id_query_title_id_max'] + ['label']
            usefeats = [f for f in chunk.columns if f not in no_use]
            input_feats = chunk[usefeats]
            input_feats = input_feats.fillna(0)

            yield [x1, x2, input_feats], y_train

def my_val_generator(file_path, batch_size):

    while True:
        
        un_training_data = pd.read_csv(file_path, chunksize=batch_size)
        
        for index_in_chunk, chunk in enumerate(un_training_data):
            # taking the values from panda into numpy array
            x1 = [[int(i) for i in v.strip().split(' ') if i in model] for v in chunk['query'].values]
            x1 = pad_sequences(x1, maxlen=5)
            x2 = [[int(i) for i in v.strip().split(' ') if i in model] for v in chunk['title'].values]
            x2 = pad_sequences(x2, maxlen=15)

            y_train = chunk['label'].values
            y_train = np_utils.to_categorical(y_train)
            
            no_use = ['query_id', 'query', 'title', 'query_query_id_nunique', 'query_id_query_title_id_max'] + ['label']
            usefeats = [f for f in chunk.columns if f not in no_use]
            input_feats = chunk[usefeats]
            input_feats = input_feats.fillna(0)

            yield [x1, x2, input_feats], y_train

checkpoint = ModelCheckpoint('./nn_model_save/model_nn_6069.h5', monitor='val_loss', save_best_only=True, verbose=2)
early_stop = EarlyStopping(monitor='val_loss', patience=1)

history = model_nn.fit_generator(my_generator(TRAIN_PATH, BATCH_SIZE),
                                 steps_per_epoch=(TRAIN_SIZE // BATCH_SIZE)+1,
                                 validation_data=my_val_generator(VALID_PATH, BATCH_SIZE),
                                 validation_steps=(VALID_SIZE // BATCH_SIZE)+1,
                                 epochs=10,
                                 verbose=1,
                                 callbacks=[checkpoint, early_stop])

def get_1e_test_part(mode):
    
    x1 = np.load('./features_final_test/test_part{}_x1.npy'.format(mode))
    x2 = np.load('./features_final_test/test_part{}_x2.npy'.format(mode))
    print(x1.shape, x2.shape)
    
    data = pd.read_hdf('./features_final_test/test_feats_part{}.h5'.format(mode))
    print(data.shape)

    return x1, x2, data

x1, x2, test_feats = get_1e_test_part(1)
preds_1 = model_nn.predict([x1, x2, test_feats],
                            batch_size=8192,
                            verbose=1)[:,1]
print(preds_1.shape)

x1, x2, test_feats = get_1e_test_part(2)
preds_2 = model_nn.predict([x1, x2, test_feats],
                            batch_size=8192,
                            verbose=1)[:,1]
print(preds_2.shape)


preds = np.append(preds_1, preds_2)
print(preds.shape)
print(np.mean(preds))

submission = pd.read_csv('/home/kesci/input/bytedance/bytedance_contest.final_2.csv', \
                    names=['query_id', 'query', 'query_title_id', 'title'],
                    usecols=['query_id','query_title_id'])
submission['pred'] = preds
submission.to_csv('./submission/sub.csv',index=False,header=None)