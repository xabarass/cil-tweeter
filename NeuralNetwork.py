import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, Activation, Input, Bidirectional
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.wrappers.scikit_learn import KerasClassifier
import config


# Making sample_weight parameter explicit in KerasClassifier.fit method
# as expected by scikit_learn using a decorator technique
def decorate_kerasClassifier_fit(fit):
    def decorated_fit(self,x,y,sample_weight=None,**kwargs):
        return fit(self,x,y,sample_weight=sample_weight,**kwargs)
    return decorated_fit

KerasClassifier.fit = decorate_kerasClassifier_fit(KerasClassifier.fit)

# Evaluation - TODO: wrap model.fit with an evaluate method


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

class ModelBuilder:
    def __init__(self, embedding_matrix, dimensions, data_set):
        self.embedding_matrix=embedding_matrix
        self.dimensions=dimensions
        self.data_set = data_set

    def register_network(self, network):
        self._network = network

    def __call__(self):
        data_set=self.data_set

        embedding_layer=None
        if self.embedding_matrix is not None:
            embedding_layer = Embedding(data_set.word_count,
                                self.dimensions,
                                weights=[self.embedding_matrix],
                                input_length=data_set.max_tweet_length,
                                trainable=False)
        else:
            embedding_layer = Embedding(data_set.word_count,
                                        self.dimensions,
                                        input_length=data_set.max_tweet_length)

        print("Max tweet length: %d"%data_set.max_tweet_length)

        print("Word count %d dimensions %d" %(data_set.word_count, self.dimensions))

        model = Sequential()
        model.add(embedding_layer)
        model.add(LSTM(150))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print(model.summary())

        model._network = self._network

        # Decorate model.fit with writer of samples ranked by weights. This decorator may potentially also be used to
        # add preprocessing to the model (to use raw tweets as x)
        def decorate_kerasSequentialfit(receiver, fit):
            def wrapped_fit(x, y, batch_size=32, epochs=10, verbose=1, callbacks=None,
                            validation_split=0., validation_data=None, shuffle=True,
                            class_weight=None, sample_weight=None, initial_epoch=0, **kwargs):
                if sample_weight is not None:
                    # TODO: get a reference to data_set and x_orig
                    def preprocessed_tweet_len(tweet):
                        return len(data_set.map_tweet_to_id_seq(
                            data_set.filter_tweet(
                                data_set.lexical_preprocessing_tweet(tweet)), config.remove_unknown_words))

                    ranked_weights = [(i[1], i[0]) for i in enumerate(sample_weight)]
                    ranked_weights.sort()
                    # with open(training_samples_sorted_by_weight.format(phase), 'a+') as tssbwf:
                    print("\n***** Training samples sorted by weight *****\n")
                    for weight, i in ranked_weights:
                        print("\t{} :\t({})\n\t\t\t{}\n\t\t\t{}\n".format(weight, y[i], data_set.x_orig[i], ' '.join(
                            [data_set.id_to_word[id] for id in x[i][-preprocessed_tweet_len(data_set.x_orig[i]):]])))

                return fit(receiver, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None,
                           validation_split=0., validation_data=None, shuffle=True,
                           class_weight=None, sample_weight=None, initial_epoch=0, **kwargs)

            return wrapped_fit

        model.fit = decorate_kerasSequentialfit(model, Sequential.fit)

        return model


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

        embedding_matrix=None
        if generate_word_embeddings:
            embedding_matrix = self._generate_word_embeddings(data_set, embedding_corpus_name)



        (x_train, y_train), (x_val, y_val)=data_set.shuffle_and_split(split_ratio)

        x_train = sequence.pad_sequences(x_train, maxlen=data_set.max_tweet_length)
        x_val = sequence.pad_sequences(x_val, maxlen=data_set.max_tweet_length)

        mBuilder= ModelBuilder(data_set=data_set, dimensions=self.dimensions, embedding_matrix=embedding_matrix)

        # Store reference to Network in mBuilder to access AdaBoost internals from Keras model
        mBuilder.register_network(self)

        model = KerasClassifier(build_fn=mBuilder, epochs=3, batch_size=64, verbose=1)

        bdt = AdaBoostClassifier(model,
                                 algorithm="SAMME.R",
                                 n_estimators=10)

        self.base_estimator = model
        self.model = bdt



        # Let's see if training is possible
        bdt.fit(x_train, y_train)

        print("***** Evaluation *****")
        for iboost, accuracy in enumerate(bdt.staged_score(x_val,y_val)):
            print( "\t{}-th boosting iteration: accuracy = {}".format(iboost,accuracy) )

        # model=mBuilder._create_model()
        #
        # evaluater=ModelEvaluater(model, x_val, y_val)
        # model.fit(x_train, y_train, epochs=4, batch_size=64, callbacks=[evaluater])
        #
        # print("Saving model...")
        # model_json = model.to_json()
        # with open(model_json_file, "w") as json_file:
        #     json_file.write(model_json)
        # model.save_weights(model_h5_file)
        # print("Saved model to disk")
        #
        # self.model=model

        print("Finished training!")

        # evaluation




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
        predictions = self.model.predict(x_test) #, batch_size=64)

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
