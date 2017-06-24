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
    def __init__(self, model, preprocessed_dataset, x_val, y_val):
        super(Callback, self).__init__()
        self.model=model
        self.x_val=x_val
        self.y_val=y_val

    def on_epoch_end(self, epoch, logs=None):
        print("\nEvaluating epoch...")
        scores = self.model.evaluate(self.x_val, self.y_val, verbose=1)
        print("\n\tValidation accuracy: %.2f%%" % (scores[1] * 100))

class ModelPredicter(Callback):
    def __init__(self, model, preprocessed_dataset, model_save_path, result_epoch_file):
        super(Callback, self).__init__()
        self.model = model
        self.preprocessed_dataset = preprocessed_dataset
        self.model_save_path=model_save_path
        self.result_epoch_file = result_epoch_file

    def on_epoch_end(self, epoch, logs=None):
        if not config.test_run:
            print("Generating prediction file for epoch %d at %s..." % (epoch, self.result_epoch_file.format(epoch)))
            save_model(self.model, self.model_save_path + "-e{}".format(epoch))

            if self.result_epoch_file is not None:
                Network.predict(self.model, self.preprocessed_dataset,
                                self.result_epoch_file.format(epoch))


# Word embeddings
class Word2VecEmbeddings:
    """Create or load word embeddings"""
    def __init__(self,
                 vocabulary, preprocessed_dataset,
                 word_embedding_dimensions, embedding_corpus_name):
        print("Generating word embeddings")
        word_embedding_model = None

        if embedding_corpus_name:
            pretrained_file = Path(embedding_corpus_name)
            if pretrained_file.is_file():
                word_embedding_model = Word2Vec.load(embedding_corpus_name)
                print("Word embeddings loaded!")

        if not word_embedding_model:
            word_embedding_model = Word2Vec(preprocessed_dataset.all_tokenized_tweets(),  # preprocessed_dataset.all_preprocessed_tweets()
                                            size=word_embedding_dimensions,
                                            window=7,
                                            min_count=vocabulary.preprocessor.min_word_occurrence,
                                            workers=8,
                                            sg=1,
                                            iter=10)

            print("Word embeddings generated!")

        if embedding_corpus_name:
            word_embedding_model.save(embedding_corpus_name)

        self.embedding_matrix = np.zeros((vocabulary.word_count, word_embedding_dimensions))
        for word, id in vocabulary.word_to_id.items():
            if word in word_embedding_model.wv.vocab:
                embedding_vector = word_embedding_model[word]
                self.embedding_matrix[id] = embedding_vector


# TODO: Implement this
class GloVeEmbeddings:
    pass


def save_model(model,model_save_path):
    model_json = model.to_json()
    with open(model_save_path + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_save_path + ".h5")
    print("Saved model to disk at %s.(json,h5)" % model_save_path)


class Network:
    word_embedding_models =  { 'word2vec' : Word2VecEmbeddings,
                               'glove'    : GloVeEmbeddings}

    @classmethod
    def create_model(cls,
                     preprocessed_dataset,
                     vocabulary,
                     word_embeddings_opt={}):

        model = Sequential()

        # Create embedding layer
        word_embeddings_opt_param = {"initializer": "word2vec", "dim": 400, "trainable": False, "corpus_name": None}
        word_embeddings_opt_param.update(word_embeddings_opt)
        if word_embeddings_opt_param["initializer"] in Network.word_embedding_models:
            word_embeddings = Network.word_embedding_models[word_embeddings_opt["initializer"]](
                vocabulary, preprocessed_dataset,
                word_embeddings_opt_param["dim"], word_embeddings_opt["corpus_name"]) # TODO: Replace explicit dict access by **word_embeddings_opt
            embedding_layer = Embedding(vocabulary.word_count,
                                        word_embeddings_opt_param["dim"],
                                        weights=[word_embeddings.embedding_matrix],
                                        input_length=preprocessed_dataset.max_tweet_length,
                                        trainable=word_embeddings_opt["trainable"])
        else:
            embedding_layer = Embedding(vocabulary.word_count,
                                        word_embeddings_opt_param["dim"],
                                        input_length=preprocessed_dataset.max_tweet_length,
                                        trainable=word_embeddings_opt["trainable"])

        print("Created Embedding layer - Word count %d, dimensions %d, max tweet length %d" %
              (vocabulary.word_count, word_embeddings_opt_param["dim"], preprocessed_dataset.max_tweet_length))

        model.add(embedding_layer)
        model.add(LSTM(200))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        print("Compiled model...")
        print(model.summary())

        return model


    @classmethod
    def train(cls,model, preprocessed_dataset,
                  training_opt={},
                  model_save_path=None,
                  result_epoch_file=None):

        training_opt_param = {"epochs":4, "batch_size":64}
        training_opt_param.update(training_opt)

        # Create training data
        (x_train, y_train, x_orig_train), (x_val, y_val, x_orig_val) = preprocessed_dataset.shuffle_and_split()

        x_train = sequence.pad_sequences(x_train, maxlen=preprocessed_dataset.max_tweet_length)
        x_val   = sequence.pad_sequences(x_val,   maxlen=preprocessed_dataset.max_tweet_length)

        evaluater=ModelEvaluater(model, preprocessed_dataset, x_val, y_val)
        predicter=ModelPredicter(model, preprocessed_dataset, model_save_path, result_epoch_file)

        model.fit(x_train, y_train, callbacks=[evaluater, predicter], **training_opt_param)

        if model_save_path is not None:
            save_model(model, model_save_path)

    @classmethod
    def output_misclassified_samples(cls,
                                     model, preprocessed_dataset, vocabulary,
                                     misclassified_samples_file=None):

        def evaluate_misclassified_samples(x, y, x_orig, phase):
            misclassified_samples = []

            x_padded = sequence.pad_sequences(x[:], maxlen=preprocessed_dataset.max_tweet_length)
            pred_y = model.predict(x_padded, batch_size=64).reshape([-1])

            for i in range(pred_y.shape[0]):
                if ((pred_y[i] > 0.5) and (y[i] == 0)) or \
                   ((pred_y[i] <= 0.5) and (y[i] == 1)):
                    misclassified_samples.append( ( 2*(pred_y[i]-0.5)*2*(y[i]-0.5), 2*(y[i]-0.5),
                                                    x_orig[i], ' '.join([vocabulary.id_to_word[id] for id in x[i]]) ) )

            misclassified_samples.sort()

            with open(misclassified_samples_file.format(phase), 'a+') as mc_s_f:
                mc_s_f.write("\n***** Misclassified {} samples *****\n".format(phase))
                for sample in misclassified_samples:
                    mc_s_f.write( "\t{} :\t({})\n\t\t\t{}\n\t\t\t{}\n".format(sample[0], sample[1], sample[2], sample[3]) )

        (x_train, y_train, x_orig_train), (x_val, y_val, x_orig_val) = preprocessed_dataset.shuffle_and_split()
        evaluate_misclassified_samples(x_val,  y_val, x_orig_val, "validation")
        evaluate_misclassified_samples(x_train, y_train, x_orig_train,"training")

    @classmethod
    def load_model(cls, model_json_file, model_h5_file):
        print("Loading model...")
        with open(model_json_file, 'r') as structure_file:
            loaded_model_json = structure_file.read()
            model = model_from_json(loaded_model_json)

        model.load_weights(model_h5_file)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print("Loaded model from disk!")

    @classmethod
    def predict(cls, model, preprocessed_dataset, prediction_file):
        if not model:
            raise Exception("You need to train or load a pretrained model in order to predict")

        x_test = sequence.pad_sequences(preprocessed_dataset.test_tweets,
                                        maxlen=preprocessed_dataset.max_tweet_length)
        predictions = model.predict(x_test, batch_size=64)

        print("Done with predictions, generating submission file...")

        with open(prediction_file, "w") as submission:
            submission.write("Id,Prediction\n")
            for i, prediction in enumerate(predictions):
                if prediction > 0.5:
                    prediction = 1
                else:
                    prediction = -1
                submission.write('%d,%d\n' % (i+1, prediction))

            print("Generated submission file (%s) with %d results" % (prediction_file,predictions.shape[0]))
