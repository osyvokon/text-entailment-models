#!/usr/bin/env python3
"""Conditional encoding model as described in "Reasoning About Entailment With
Neural Attention" (Rocktäschel, 2016).
"""

import os
import numpy as np
import pytest
from unittest.mock import Mock
from gensim.models import KeyedVectors
from keras.layers import Embedding, Input, Dense, LSTM
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


class ConditionalEncodingModel:
    """Conditional encoding model as described in (Rocktäschel, 2016).

    Args:
        word2vec (str): path to the word2vec binary file.
        debug (bool): if True, run faster and produce more output.
        premise_maxlen (int): max number of words in a premise
        premise_k (int): number of premise LSTM units.
        premise_dropout (float): premise LSTM dropout rate.
        hyp_maxlen (int): max number of words in a hypothesis
        hyp_k (int): number of hyp LSTM units.
        hyp_dropout (float): hyp LSTM dropout rate.
        seed (int): random generator seed.
    """

    def __init__(self, word2vec, debug=False, **kwargs):
        self._word2vec = word2vec
        self._vocab = None
        self._word2vec_loaded = False
        self.debug = debug
        self.maxlen = kwargs.pop('maxlen', 20)
        self.premise_maxlen = kwargs.pop('premise_maxlen', 20)
        self.premise_k = kwargs.pop('premise_k', 100)
        self.premise_dropout = kwargs.pop('premise_dropout', 0.1)
        self.hyp_maxlen = kwargs.pop('hyp_maxlen', 20)
        self.hyp_k = kwargs.pop('hyp_k', 100)
        self.hyp_dropout = kwargs.pop('hyp_dropout', 0.1)

        self._model = None
        self._rng = np.random.RandomState(kwargs.pop('seed', 0))

    def build(self):

        # Encode premise by the first LSTM
        premise_input = Input(shape=(self.premise_maxlen, ), dtype='int32', name='premise_input')
        premise_embed = Embedding(self.word2vec.shape[0],
                                  self.word2vec.shape[1],
                                  weights=[self.word2vec],
                                  trainable=False)(premise_input)
        premise_lstm = LSTM(self.premise_k)  #, dropout=self.premise_dropout)
        premise_encoded = premise_lstm(premise_embed)

        # Encode hypothesis by another LSTM
        hyp_input = Input(shape=(self.hyp_maxlen, ), dtype='int32', name='hyp_input')
        hyp_embed = Embedding(self.word2vec.shape[0],
                              self.word2vec.shape[1],
                              weights=[self.word2vec],
                              trainable=False)(hyp_input)
        hyp_lstm = LSTM(self.hyp_k) #, dropout=self.hyp_dropout)
        hyp_encoded = hyp_lstm(hyp_embed)

        # Make a prediction from the last output vector
        NUM_CLASSES = 3
        predictions = Dense(NUM_CLASSES, activation='softmax')(hyp_encoded)

        # Compile the model
        model = Model(inputs=[premise_input, hyp_input], outputs=predictions)
        return model

    @property
    def model(self):
        if self._model is None:
            self._model = self.build()
        return self._model

    def fit(self, data):
        """Train the model.

        Args:
            data: training data, list of (label, premise, hypothesis) tuples,
                where `label` is a string, one of 'neutral', 'entailment',
                'contradiction', `premise` and `hypothesis` are tokenized
                premise sentences (each is a list of string tokens).
        """

        # Prepare data
        labels = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
        Y = to_categorical([labels[y] for y, *_ in data])
        premise = [self.vectorize(p) for _, p, h in data]
        hyp = [self.vectorize(h) for _, p, h in data]

        premise = pad_sequences(premise, maxlen=self.premise_maxlen, value=0, padding='pre')
        hyp = pad_sequences(hyp, maxlen=self.hyp_maxlen, value=0, padding='post')

        model = self.model
        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        checkpoint = ModelCheckpoint('model.check', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=2, min_lr=0.0001)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      callbacks=[reduce_lr, early_stopping, checkpoint])
        history = model.fit([premise, hyp], Y,
                            validation_split=0.25,
                            epochs=3)
        return history

    def save(self, path):
        model.save(path)

    def vectorize(self, sentence):
        V = {word: index for index, word in enumerate(self.vocab)}
        assert '<unk>' in V
        result = np.empty_like(sentence, dtype=np.int32)
        for i, token in enumerate(sentence):
            if token not in V:
                token = '<unk>'
            result[i] = V[token]
        return result

    #def prepare_X_data(self, premise, hypothesis):

    @property
    def word2vec(self):
        """Lazily load word2vec embeddings. """
        if not self._word2vec_loaded:
            self._load_word2vec()
            self._word2vec_loaded = True
        return self._word2vec

    @property
    def vocab(self):
        """List of known words. """
        if not self._word2vec_loaded:
            self._load_word2vec()
            self._word2vec_loaded = True
        return self._vocab

    def _load_word2vec(self):
        if isinstance(self._word2vec, str):
            print("Loading word2vec... ", end='', flush=True)
            limit = 10000 if self.debug else None
            vecs = KeyedVectors.load_word2vec_format(
                self._word2vec, binary=True, limit=limit)
        else:
            vecs = self._word2vec

        self._word2vec = vecs.syn0
        self._vocab = vecs.index2word

        # Make sure we have a random <unk> vector
        if not '<unk>' in self._vocab:
            dim = vecs.syn0.shape[1]
            random_vec = self.rng.randn(dim)
            self._vocab.append('<unk>')
            self._word2vec = np.vstack((vecs.syn0, random_vec))
        print("done")


class Test_ConditionalEncodingModel:

    def test_vectorize(self, word2vec):
        word2vec.index2word = ['<unk>', 'I', 'go', 'to', 'school']
        model = ConditionalEncodingModel(word2vec)
        sentence = 'I go to fancy school'.split()
        expected = [1, 2, 3, 0,   4]
        assert model.vectorize(sentence).tolist() == expected

    def test_fit(self, word2vec, rng):
        vocab = ['<unk>', 'I', 'go', 'to', 'school']
        word2vec.index2word = vocab
        word2vec.syn0 = rng.randn(len(vocab), 300)
        data = [
            ["entailment", "I go to school".split(), "There is a school".split()],
            ["neutral", "I went to school".split(), "School is good".split()],
        ]
        model = ConditionalEncodingModel(word2vec)
        history = model.fit(data)
        assert history.history['loss'][0] > history.history['loss'][-1]

    @pytest.fixture
    def word2vec(self):
        return Mock(spec=KeyedVectors).return_value
    
    @pytest.fixture
    def rng(self):
        np.random.seed(42)
        return np.random.RandomState(42)


def read_tsv(path):
    return [line.split('\t') for line in open(path).read().split('\n')]


if __name__ == '__main__':
    model = ConditionalEncodingModel(
        word2vec="/Users/oleksiy.syvokon/data/word2vec/GoogleNews-vectors-negative300.bin")

    here = os.path.dirname(__file__)
    data = os.path.join(here, '../data/')
    path = os.path.join(data, 'model_two_lstm.keras')
    dataset = read_tsv(os.path.join(data, 'train.txt'))

    model.build()
    model.fit(dataset)
    model.save(path)
