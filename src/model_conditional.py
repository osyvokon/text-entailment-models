#!/usr/bin/env python3
"""Conditional encoding model as described in "Reasoning About Entailment With
Neural Attention" (Rocktäschel, 2016).

#############################################
# WARNING
#
# FOR NOW, THIS ONLY IMPLEMENTS BOWMAN 2015!
# 
#############################################

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
        vocab_limit (int): if set, use only that many top words.
    """

    def __init__(self, word2vec, debug=False, **kwargs):
        self._word2vec = word2vec
        self._vocab = None
        self._vocab_index = None
        self._word2vec_loaded = False
        self.debug = debug
        self.maxlen = kwargs.pop('maxlen', 20)
        self.premise_maxlen = kwargs.pop('premise_maxlen', 20)
        self.premise_k = kwargs.pop('premise_k', 100)
        self.premise_dropout = kwargs.pop('premise_dropout', 0.1)
        self.hyp_maxlen = kwargs.pop('hyp_maxlen', 20)
        self.hyp_k = kwargs.pop('hyp_k', 100)
        self.hyp_dropout = kwargs.pop('hyp_dropout', 0.1)
        self.vocab_limit = kwargs.pop('vocab_limit', None)

        self._model = None
        self._rng = np.random.RandomState(kwargs.pop('seed', 0))

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

        self._log_start('Vectorize data... ')
        labels = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
        Y = to_categorical([labels[y] for y, *_ in data])

        premise = [self.vectorize(p) for _, p, h in data]
        hyp = [self.vectorize(h) for _, p, h in data]
        premise = pad_sequences(premise, maxlen=self.premise_maxlen, value=0, padding='pre')
        hyp = pad_sequences(hyp, maxlen=self.hyp_maxlen, value=0, padding='post')
        self._log_done()

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

        self._log_start('Vectorize data... ')
        premise = [self.vectorize(p) for p, h in data]
        hyp = [self.vectorize(h) for p, h in data]
        premise = pad_sequences(premise, maxlen=self.premise_maxlen, value=0, padding='pre')
        hyp = pad_sequences(hyp, maxlen=self.hyp_maxlen, value=0, padding='post')
        self._log_done()

        predictions = self.model.predict([premise, hyp])
        labels = ['neutral', 'contradiction', 'entailment']
        result = [labels[i] for i in predictions.argmax(axis=1)]
        return result

    def save(self, path):
        self._log_start("Saving model to {}".format(path))
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

    #def prepare_X_data(self, premise, hypothesis):

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


class Test_ConditionalEncodingModel:

    def test_vectorize(self, model):
        sentence = 'I go to fancy school'.split()
        expected = [1, 2, 3, 0,   4]
        assert model.vectorize(sentence).tolist() == expected

    def test_fit(self, model, dataset):
        history = model.fit(dataset)
        assert history.history['loss'][0] > history.history['loss'][-1]

    def test_predict(self, model, dataset):
        model.fit(dataset)

        data = [["I go to school".split(), "School exists".split()]]
        predictions = model.predict(data)
        assert predictions == ['entailment']

    @pytest.fixture
    def model(self, word2vec, rng):
        vocab = ['<unk>', 'I', 'go', 'to', 'school']
        word2vec.index2word = vocab
        word2vec.syn0 = rng.randn(len(vocab), 300)
        return ConditionalEncodingModel(word2vec)

    @pytest.fixture
    def dataset(self):
        return [
            ["entailment", "I go to school".split(), "There is a school".split()],
            ["neutral", "I went to school".split(), "School is good".split()],
        ]

    @pytest.fixture
    def word2vec(self):
        return Mock(spec=KeyedVectors).return_value
    
    @pytest.fixture
    def rng(self):
        np.random.seed(42)
        return np.random.RandomState(42)


def read_tsv(path):
    return [line.split('\t') for line in open(path).read().split('\n') if line]

def read_dataset(path):
    dataset = []
    for label, premise, hyp in read_tsv(path):
        dataset.append((label, premise.split(), hyp.split()))
    return dataset


def cmd_train(args):
    dataset = read_dataset(args.data)
    model = ConditionalEncodingModel(word2vec=args.word2vec,
                                     vocab_limit=args.vocab_limit,
                                     )
    model.fit(dataset, epochs=args.epochs)
    model.save(args.output)


if __name__ == '__main__':
    here = os.path.dirname(__file__)
    data = os.path.join(here, '../data/')

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers()
    sub = subparsers.add_parser('train', help="Train model")
    sub.set_defaults(func=cmd_train)
    sub.add_argument('output', help='File to save trained model')
    sub.add_argument('--epochs', help='Number of epochs. Default is %(default)s',
                     default=30, type=int)
    sub.add_argument('--data', default=os.path.join(data, 'train.txt'),
                        help='Path to the training data. Default is %(default)s')
    sub.add_argument('--word2vec', default=os.path.join(data, 'word2vec.bin'),
                        help='Word2vec binary file. Default is %(default)s')
    sub.add_argument('--vocab-limit', default=None, type=int,
                        help='Use that many most popular words')

    args = parser.parse_args()
    if args.func:
        args.func(args)
    else:
        parser.print_help()
