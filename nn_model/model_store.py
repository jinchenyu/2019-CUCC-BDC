#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, LSTM, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate, Lambda
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, Multiply, Subtract, Dot, Softmax
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras import backend as K
from keras.layers.advanced_activations import PReLU


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim



def m_bilstm_attention_esim_add_feature_model(max_features, max_len_1, max_len_2, feature_size, embedding_matrix):
    # 1. lstm attention
    inp1 = Input(shape=(max_len_1,))
    x1 = Embedding(max_features, 200, weights=[embedding_matrix], trainable=False)(inp1)
    x1 = Bidirectional(LSTM(40, return_sequences=True))(x1)
    atten_x1 = Attention(max_len_1)(x1)

    inp2 = Input(shape=(max_len_2,))
    x2 = Embedding(max_features, 200, weights=[embedding_matrix], trainable=False)(inp2)
    x2 = Bidirectional(LSTM(40, return_sequences=True))(x2)
    atten_x2 = Attention(max_len_2)(x2)

    avg_pool_1 = GlobalAveragePooling1D()(x1)
    max_pool_1 = GlobalMaxPooling1D()(x1)   

    avg_pool_2 = GlobalAveragePooling1D()(x2)
    max_pool_2 = GlobalMaxPooling1D()(x2)  

    # 2. ESIM
    e = Dot(axes=2)([x1, x2])
    e1 = Softmax(axis=2)(e)
    e2 = Softmax(axis=1)(e)
    e1 = Lambda(K.expand_dims, arguments={'axis' : 3})(e1)
    e2 = Lambda(K.expand_dims, arguments={'axis' : 3})(e2)

    _x1 = Lambda(K.expand_dims, arguments={'axis' : 1})(x2)
    _x1 = Multiply()([e1, _x1])
    _x1 = Lambda(K.sum, arguments={'axis' : 2})(_x1)
    _x2 = Lambda(K.expand_dims, arguments={'axis' : 2})(x1)
    _x2 = Multiply()([e2, _x2])
    _x2 = Lambda(K.sum, arguments={'axis' : 1})(_x2)

    m1 = Concatenate()([x1, _x1, Subtract()([x1, _x1]), Multiply()([x1, _x1])])
    m2 = Concatenate()([x2, _x2, Subtract()([x2, _x2]), Multiply()([x2, _x2])])

    y1 = Bidirectional(LSTM(40, return_sequences=True))(m1)
    y2 = Bidirectional(LSTM(40, return_sequences=True))(m2)

    mx1 = Lambda(K.max, arguments={'axis' : 1})(y1)
    av1 = Lambda(K.mean, arguments={'axis' : 1})(y1)
    mx2 = Lambda(K.max, arguments={'axis' : 1})(y2)
    av2 = Lambda(K.mean, arguments={'axis' : 1})(y2)

    # 3. feats

    inp3 = Input(shape=(feature_size,))
    # x3 = BatchNormalization()(inp3)
    x3 = Dense(256, activation="relu")(inp3)

    # 4. concat

    conc = concatenate([atten_x1, avg_pool_1, max_pool_1, atten_x2, avg_pool_2, max_pool_2, mx1, av1, mx2, av2, x3])
    # conc = concatenate([avg_pool_1, avg_pool_2])
    conc = Dropout(0.2)(conc)
    conc = BatchNormalization()(conc)

    conc = Dense(256, activation="relu")(conc)
    conc = Dropout(0.2)(conc)
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.2)(conc)
    outp = Dense(2, activation="sigmoid")(conc)

    model_nn = Model(inputs=[inp1, inp2, inp3], outputs=outp)
    model_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model_nn
