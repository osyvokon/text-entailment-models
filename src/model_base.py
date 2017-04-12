#!/usr/bin/env python3
"""Base class for text entailment models. It handles things like
input vectorization and embedding.

"""

import argparse
import os
import numpy as np
import pytest
from unittest.mock import Mock
from gensim.models import KeyedVectors
from keras.layers import Embedding, Input, Dense, LSTM, concatenate, Dropout
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


class BaseModel:
    """

    Args:
        word2vec (str): path to the word2vec binary file.
        debug (bool): if True, run faster and produce more output.
        seed (int): random generator seed.
    """

    def __init__(self, word2vec, debug=False, **kwargs):
        self._word2vec = word2vec
        self._vocab = None
        self._vocab_index = None
        self._word2vec_loaded = False
        self.debug = debug
        self.vocab_limit = kwargs.pop('vocab_limit', None)
        self._rng = np.random.RandomState(kwargs.pop('seed', 0))

        self._model = None

    def build(self):

        # Encode premise by the first LSTM
        premise_input = Input(shape=(self.premise_maxlen, ), dtype='int32', name='premise_input')
        premise_embed = Embedding(self.word2vec.shape[0],
                                  self.word2vec.shape[1],
                                  weights=[self.word2vec],
                                  trainable=False)(premise_input)
        premise_dropout = Dropout(self.premise_dropout)(premise_embed)
        premise_lstm = LSTM(self.premise_k)
        premise_encoded = premise_lstm(premise_dropout)

        # Encode hypothesis by another LSTM
        hyp_input = Input(shape=(self.hyp_maxlen, ), dtype='int32', name='hyp_input')
        hyp_embed = Embedding(self.word2vec.shape[0],
                              self.word2vec.shape[1],
                              weights=[self.word2vec],
                              trainable=False)(hyp_input)
        hyp_dropout = Dropout(self.hyp_dropout)(hyp_embed)
        hyp_lstm = LSTM(self.hyp_k)
        hyp_encoded = hyp_lstm(hyp_dropout)

        # Make a prediction from the last output vector
        NUM_CLASSES = 3
        merged = concatenate([premise_encoded, hyp_encoded], axis=-1)
        predictions = Dense(NUM_CLASSES, activation='softmax')(merged)

        # Compile the model
        model = Model(inputs=[premise_input, hyp_input], outputs=predictions)
        return model

    @property
    def model(self):
        if self._model is None:
            self._model = self.build()
        return self._model

    def fit(self, data, **kwargs):
        """Train the model.

        Args:
            data: training data, list of (label, premise, hypothesis) tuples,
                where `label` is a string, one of 'neutral', 'entailment',
                'contradiction', `premise` and `hypothesis` are tokenized
                premise sentences (each is a list of string tokens).
            epochs (int, optional): number of epochs

        Returns:
            keras.history object

        """

        epochs = kwargs.pop('epochs', 3)
        self._load_word2vec()

        Y, hyp, premise = self.vectorize_training_data(data)

        model = self.model
        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        checkpoint = ModelCheckpoint('/tmp/model.check', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=2, min_lr=0.0001)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit([premise, hyp], Y,
                            validation_split=0.25,
                            epochs=epochs,
                            callbacks=[reduce_lr, early_stopping, checkpoint])
        return history

    def predict(self, data):
        """Make predictions on the unseen data.

        Args:
            data: list of (premise, hypothesis) tuples

        Returns:
            list of string labels ('neutral', 'contradiction', or 'entailment')
        """

        premise, hyp = self.vectorize_test_data(data)
        predictions = self.model.predict([premise, hyp])
        labels = ['neutral', 'contradiction', 'entailment']
        result = [labels[i] for i in predictions.argmax(axis=1)]
        return result

    def save(self, path):
        self._log_start("Saving model to {}... ".format(path))
        self.model.save(path)
        self._log_done()

    def vectorize(self, sentence):
        V = self.vocab_index
        assert '<unk>' in V
        result = np.empty_like(sentence, dtype=np.int32)
        for i, token in enumerate(sentence):
            if token not in V:
                token = '<unk>'
            result[i] = V[token]
        return result

    def vectorize_training_data(self, data):
        self._log_start('Vectorize data... ')
        labels = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
        Y = to_categorical([labels[y] for y, *_ in data])
        premise = [self.vectorize(p) for _, p, h in data]
        hyp = [self.vectorize(h) for _, p, h in data]

        premise = pad_sequences(premise, maxlen=self.premise_maxlen, value=0, padding='pre')
        hyp = pad_sequences(hyp, maxlen=self.hyp_maxlen, value=0, padding='post')

        self._log_done()
        return Y, premise, hyp

    def vectorize_test_data(self, data):
        self._log_start('Vectorize data... ')
        premise = [self.vectorize(p) for p, h in data]
        hyp = [self.vectorize(h) for p, h in data]
        premise = pad_sequences(premise, maxlen=self.premise_maxlen, value=0, padding='pre')
        hyp = pad_sequences(hyp, maxlen=self.hyp_maxlen, value=0, padding='post')

        self._log_done()
        return premise, hyp

    @property
    def word2vec(self):
        """Lazily load word2vec embeddings. """

        self._load_word2vec()
        return self._word2vec

    @property
    def vocab(self):
        """List of known words. """

        self._load_word2vec()
        return self._vocab
    
    @property
    def vocab_index(self):
        """{word: index} mapping. """

        self._load_word2vec()
        return self._vocab_index

    def _load_word2vec(self):
        if self._word2vec_loaded:
            return

        if isinstance(self._word2vec, str):
            self._log_start("Loading word2vec... ")
            limit = 10000 if self.debug else self.vocab_limit
            vecs = KeyedVectors.load_word2vec_format(
                self._word2vec, binary=True, limit=limit)
        else:
            vecs = self._word2vec

        self._word2vec = vecs.syn0
        self._vocab = vecs.index2word

        # Make sure we have a random <unk> vector
        if not '<unk>' in self._vocab:
            dim = vecs.syn0.shape[1]
            random_vec = self._rng.randn(dim)
            self._vocab.append('<unk>')
            self._word2vec = np.vstack((vecs.syn0, random_vec))

        self._vocab_index = {word: index for index, word in enumerate(self._vocab)}
        self._word2vec_loaded = True
        self._log_done()

    def _log_start(self, msg):
        print(msg, end='', flush=True)
    
    def _log_done(self):
        print("done")


def read_tsv(path):
    return [line.split('\t') for line in open(path).read().split('\n') if line]


def read_dataset(path):
    dataset = []
    for label, premise, hyp in read_tsv(path):
        dataset.append((label, premise.split(), hyp.split()))
    return dataset
