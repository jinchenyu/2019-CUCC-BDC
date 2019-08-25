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

######################## 3、train
checkpoint = ModelCheckpoint('./nn_model_save/model_nn_3kw_605model.h5', monitor='val_loss', save_best_only=True, verbose=2)
early_stop = EarlyStopping(monitor='val_loss', patience=1)

# load train data
x1, x2, train_feats = load_train_data()

history = model_nn.fit([x1, x2, train_feats],
                          y_train,
                        #  validation_data=([valid_x1,valid_x2,valid_x3],y_valid),
                          validation_split=0.1,
                          batch_size=4096,
                          epochs=20,
                          verbose=1,
                          callbacks=[checkpoint, early_stop]
                         )

######################## 4、predict

test_x1, test_x2, test_feats = get_1e_test_part(1) # mode=1, part1

predict_part1 = model_nn.predict([test_x1, test_x2, test_feats],
                                batch_size=4096,
                                verbose=1
                                )

test_x1, test_x2, test_feats = get_1e_test_part(2) # mode=1, part1

predict_part2 = model_nn.predict([test_x1, test_x2, test_feats],
                                batch_size=4096,
                                verbose=1
                                )

######################## submission
preds = np.append(predict_part1, predict_part2)
print(preds.shape)
print(np.mean(preds))

submission = pd.read_csv('/home/kesci/input/bytedance/bytedance_contest.final_2.csv', \
                    names=['query_id', 'query', 'query_title_id', 'title'],
                    usecols=['query_id','query_title_id'])
submission['pred'] = preds
submission.to_csv('./submission/sub.csv',index=False,header=None)