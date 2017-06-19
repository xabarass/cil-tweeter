import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, Activation, Input
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPooling1D
from keras.preprocessing import sequence
from keras.layers.merge import Concatenate
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from gensim.models import Word2Vec
from pathlib import Path
import os
import config

class ModelEvaluater(Callback):
    def __init__(self, model, x_val, y_val):
        super(Callback, self).__init__()
        self.model=model
        self.x_val=x_val
        self.y_val=y_val

    def on_epoch_end(self, epoch, logs=None):
        print()
        print("Evaluating epoch...")
        scores = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        print("Saving model...")
        model_json = self.model.to_json()
        with open("model-e{}.json".format(epoch), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model-e{}.h5".format(epoch))
        print("Saved model to disk")

class Network:
    def __init__(self, input_dimension):
        self.dimensions=input_dimension

    def _generate_word_embeddings(self, data_set, embedding_corpus_name):
        print("Generating word embeddings")
        word_embedding_model=None
        if embedding_corpus_name:
            pretrained_file=Path(embedding_corpus_name)
            if pretrained_file.is_file():
                word_embedding_model=Word2Vec.load(embedding_corpus_name)
                print("Word embeddings loaded!")

        if not word_embedding_model:
            word_embedding_model = Word2Vec(data_set.full_tweets,
                                            size=self.dimensions,
                                            window=7,
                                            min_count=data_set.min_word_occurence,
                                            workers=8,
                                            sg=1,
                                            iter=10)
            print("Word embeddings generated!")

        if embedding_corpus_name:
            word_embedding_model.save(embedding_corpus_name)

        embedding_matrix = np.zeros((data_set.word_count, self.dimensions))
        for word, id in data_set.word_to_id.items():
            if word in word_embedding_model.wv.vocab:
                embedding_vector=word_embedding_model[word]
                embedding_matrix[id]=embedding_vector

        return embedding_matrix

    def train(self, data_set, split_ratio, 
              generate_word_embeddings=False, embedding_corpus_name=None,
              model_json_file=config.model_json, model_h5_file=config.model_h5):
        embedding_layer=None
        if generate_word_embeddings:
            embedding_matrix=self._generate_word_embeddings(data_set, embedding_corpus_name)
            embedding_layer = Embedding(data_set.word_count,
                                self.dimensions,
                                weights=[embedding_matrix],
                                input_length=data_set.max_tweet_length,
                                trainable=False)
        else:
            embedding_layer = Embedding(data_set.word_count,
                                        self.dimensions,
                                        input_length=data_set.max_tweet_length)

        print("Max tweet length: %d"%data_set.max_tweet_length)

        (x_train, y_train), (x_val, y_val)=data_set.shuffle_and_split(split_ratio)

        x_train = sequence.pad_sequences(x_train, maxlen=data_set.max_tweet_length)
        x_val = sequence.pad_sequences(x_val, maxlen=data_set.max_tweet_length)

        print("Word count %d dimensions %d" %(data_set.word_count, self.dimensions))

        model = Sequential()
        model.add(embedding_layer)
        model.add(LSTM(300))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print(model.summary())

        evaluater=ModelEvaluater(model, x_val, y_val)
        model.fit(x_train, y_train, epochs=4, batch_size=64, callbacks=[evaluater])

        print("Saving model...")
        model_json = model.to_json()
        with open(model_json_file, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(model_h5_file)
        print("Saved model to disk")

        self.model=model

    def load_model(self, structure, params):
        print("Loading model...")
        with open(structure, 'r') as structure_file:
            loaded_model_json = structure_file.read()
            self.model = model_from_json(loaded_model_json)

        self.model.load_weights(params)

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print("Loaded model from disk!")

    def predict(self, data_set, prediction_file):
        if not self.model:
            raise Exception("You need to train or load pretrained model in order to predict")

        x_test = sequence.pad_sequences(data_set.test_tweets, maxlen=data_set.max_tweet_length)
        predictions = self.model.predict(x_test, batch_size=64)

        print("Done with predictions, generating submission file...")

        with open(prediction_file, "w") as submission:
            submission.write("Id,Prediction\n")
            i = 1
            for prediction in predictions:
                if prediction > 0.5:
                    prediction = 1
                else:
                    prediction = -1

                submission.write('%d,%d\n' % (i, prediction))
                i = i + 1

            print("Generated submission file (%s) with %d results" % (prediction_file,(i - 1)))
