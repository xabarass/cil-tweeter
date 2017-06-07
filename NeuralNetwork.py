import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, Activation, Input
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.merge import Concatenate
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import Bidirectional
from keras.models import Model
import os

class Network:
    def __init__(self, input_dimension):
        self.dimensions=input_dimension

    @DeprecationWarning
    def _load_pretrained_glove(self, glove_file_dir, data_set):
        glove_file = "glove.twitter.27B.%dd.txt" % self.dimensions
        glove_path=os.path.join(glove_file_dir,glove_file)
        embeddings = {}

        with open(glove_path,'r') as emb:
            for line in emb:
                tokens=line.split(' ')
                word=tokens[0]
                vector=np.asarray(tokens[1:], dtype='float32')
                embeddings[word]=vector

        found_replacements=0
        embedding_matrix = np.zeros((data_set.word_count, self.dimensions))
        for word in data_set.unused_words:
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[0] = embedding_vector
                found_replacements=found_replacements+1

        print("Found replacements in other dictionary %d" %found_replacements)
        return embedding_matrix

    def train(self, data_set, split_ratio):
        # embedding_matrix=self._load_pretrained_glove('/tmp',data_set)
        # embedding_layer1 = Embedding(data_set.word_count,
        #                     self.dimensions,
        #                     weights=[embedding_matrix],
        #                     input_length=data_set.max_tweet_length,
        #                     trainable=False)

        (x_train, y_train), (x_val, y_val)=data_set.shuffle_and_split(split_ratio)

        x_train = sequence.pad_sequences(x_train, maxlen=data_set.max_tweet_length)
        x_val = sequence.pad_sequences(x_val, maxlen=data_set.max_tweet_length)

        embedding_layer=Embedding(data_set.word_count, self.dimensions, input_length=data_set.max_tweet_length)

        model = Sequential()
        model.add(embedding_layer)
        model.add(Convolution1D(64,
                                3,
                                padding='causal',
                                activation='relu',
                                strides=1))
        model.add(Dropout(0.25))
        model.add(Bidirectional(LSTM(80)))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print(model.summary())

        model.fit(x_train, y_train, epochs=2, batch_size=64)

        scores = model.evaluate(x_val, y_val, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        print("Saving model...")
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")