#!/usr/bin/env python3
"""Model that loosely follows [Bowman, 2015]. """

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

from model_base import BaseModel, read_dataset


class BowmanEncodingModel(BaseModel):
    """Model that loosely follows [Bowman, 2015].

    An LSTM encodes premise and hypothesis. The resulting
    representations are then fed to a dense layer that makes a
    classification.

    Args:
        word2vec (str): path to the word2vec binary file.
        debug (bool): if True, run faster and produce more output.
        k (int): number of LSTM units.
        premise_maxlen (int): max number of words in a premise
        premise_dropout (float): premise LSTM dropout rate.
        hyp_maxlen (int): max number of words in a hypothesis
        hyp_dropout (float): hyp LSTM dropout rate.
        seed (int): random generator seed.
        vocab_limit (int): if set, use only that many top words.
    """

    def __init__(self, word2vec, debug=False, **kwargs):
        super().__init__(word2vec, debug, **kwargs)

        self.k = kwargs.pop('k', 100)
        self.premise_maxlen = kwargs.pop('premise_maxlen', 20)
        self.premise_dropout = kwargs.pop('premise_dropout', 0.1)
        self.hyp_maxlen = kwargs.pop('hyp_maxlen', 20)
        self.hyp_k = kwargs.pop('hyp_k', 100)
        self.hyp_dropout = kwargs.pop('hyp_dropout', 0.1)
        self.vocab_limit = kwargs.pop('vocab_limit', None)

    def build(self):

        # Prepare premise input
        premise_input = Input(shape=(self.premise_maxlen, ), dtype='int32', name='premise_input')
        premise_embed = Embedding(self.word2vec.shape[0],
                                  self.word2vec.shape[1],
                                  weights=[self.word2vec],
                                  trainable=False)(premise_input)
        premise_dropout = Dropout(self.premise_dropout)(premise_embed)

        # Prepare hypothesis input
        hyp_input = Input(shape=(self.hyp_maxlen, ), dtype='int32', name='hyp_input')
        hyp_embed = Embedding(self.word2vec.shape[0],
                              self.word2vec.shape[1],
                              weights=[self.word2vec],
                              trainable=False)(hyp_input)
        hyp_dropout = Dropout(self.hyp_dropout)(hyp_embed)

        # Encode premise and hypothesis by the LSTM
        lstm = LSTM(self.k)
        premise_encoded = lstm(premise_dropout)
        hyp_encoded = lstm(hyp_dropout)

        # Make a prediction from the last output vector
        NUM_CLASSES = 3
        merged = concatenate([premise_encoded, hyp_encoded], axis=-1)
        predictions = Dense(NUM_CLASSES, activation='softmax')(merged)

        # Compile the model
        model = Model(inputs=[premise_input, hyp_input], outputs=predictions)
        return model


class Test_BowmanEncodingModel:

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
        return BowmanEncodingModel(word2vec)

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


def cmd_train(args):
    dataset = read_dataset(args.data)
    model = BowmanEncodingModel(word2vec=args.word2vec,
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
