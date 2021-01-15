# import tensorflow
import numpy as np
import os
import pickle

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential


class ChatBotModel:
    _model = None
    _tokenizer = None
    _total_words = None
    _max_sequence_len = None
    _input_sequences = None

    @staticmethod
    def load_data(file_path):
        with open(file_path, 'rb') as f:
            conversations = pickle.load(f, encoding='utf-8')

        return conversations

    def define(self, corpus: list):

        if corpus is not None:
            self._tokenizer = Tokenizer()
            self._tokenizer.fit_on_texts(corpus)
            self._total_words = len(self._tokenizer.word_index) + 1

            self._input_sequences = []
            for line in corpus:
                token_list = self._tokenizer.texts_to_sequences([line])[0]
                for i in range(1, len(token_list)):
                    n_gram_sequence = token_list[:i + 1]
                    self._input_sequences.append(n_gram_sequence)

            # pad sequences
            self._max_sequence_len = max([len(x) for x in self._input_sequences])

        # define layers
        self._model = Sequential()
        self._model.add(Embedding(self._total_words, 64, input_length=self._max_sequence_len - 1))
        self._model.add(Bidirectional(LSTM(20)))
        self._model.add(Dense(self._total_words, activation='softmax'))
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return

    def fit(self):
        input_sequences = np.array(pad_sequences(self._input_sequences, maxlen=self._max_sequence_len, padding='pre'))

        # create predictors and label
        xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
        ys = tf.keras.utils.to_categorical(labels, num_classes=self._total_words)

        # callback
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy', min_delta=0.005, patience=10, verbose=0, restore_best_weights=True
        )

        # fit model
        self._model.fit(xs, ys, epochs=300, verbose=1, callbacks=[early_stop_callback])
        return

    def save_weights_model_info(self, model_name):
        os.makedirs(f'../data/models/{model_name}/', exist_ok=True)

        self._model.save_weights(f'../data/models/{model_name}/model_weights')
        with open(f'../data/models/{model_name}/model_info.pkl', 'wb') as f:
            pickle.dump({'tokenizer': self._tokenizer,
                         'max_sequence_len': self._max_sequence_len,
                         'total_words': self._total_words}, f)
        return

    def load_weights(self, model_name):
        self._model.load_weights(f'../data/models/{model_name}/model_weights')

        return

    def load_model_info(self, model_name):
        with open(f'../data/models/{model_name}/model_info.pkl', 'rb') as f:
            info_model = pickle.load(f)

        self._tokenizer = info_model['tokenizer']
        self._max_sequence_len = info_model['max_sequence_len']
        self._total_words = info_model['total_words']

        return

    def predict(self, seed_texts: list = None, next_words: int = 2):
        if seed_texts is None:
            seed_texts = ['te']

        predicted_texts = []
        for seed_text in seed_texts:
            original_text = seed_text
            for _ in range(next_words):
                token_list = self._tokenizer.texts_to_sequences([seed_text])[0]
                token_list = pad_sequences([token_list], maxlen=self._max_sequence_len - 1, padding='pre')
                predicted = np.argmax(self._model.predict(token_list), axis=-1)
                output_word = ""
                for word, index in self._tokenizer.word_index.items():
                    if index == predicted:
                        output_word = word
                        break
                seed_text += " " + output_word
            predicted_texts.append(seed_text)
            print(f'{original_text} -> {seed_text}')

        return predicted_texts
