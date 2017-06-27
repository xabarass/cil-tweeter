import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, Activation, Input, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPooling1D
from keras.preprocessing import sequence
from keras.layers.merge import Concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
from keras.callbacks import Callback

import Models
from Emailer import Emailer
import config

# import our own modules
from WordEmbeddings import Word2VecEmbeddings, GloVeEmbeddings
from KerasUtils import save_model


# Callbacks for logging during training
class ModelEvaluater(Callback):
    def __init__(self, model, preprocessed_dataset, x_val, y_val):
        super(Callback, self).__init__()
        self.model=model
        self.x_val=x_val
        self.y_val=y_val
        if hasattr(config,'email'):
            self.emailer=Emailer('Training network update', config.email)
        else:
            self.emailer=None

    def on_epoch_end(self, epoch, logs=None):
        print("\nEvaluating epoch...")
        scores = self.model.evaluate(self.x_val, self.y_val, verbose=1)
        print("\n\tValidation accuracy: %.2f%%" % (scores[1] * 100))
        if self.emailer is not None:
            try:
                self.emailer.send_report("Epoch {} has score {}% on validation".format(epoch,scores[1]*100));
            except Exception:
                print("Error sending email!")


class ModelPredicter(Callback):
    def __init__(self, model, preprocessed_dataset, model_save_path, result_epoch_file):
        super(Callback, self).__init__()
        self.model = model
        self.preprocessed_dataset = preprocessed_dataset
        self.model_save_path=model_save_path
        self.result_epoch_file = result_epoch_file

    def on_epoch_end(self, epoch, logs=None):
        print("Generating prediction file for epoch %d at %s..." % (epoch, self.result_epoch_file.format(epoch)))
        save_model(self.model, self.model_save_path + "-e{}".format(epoch))

        if self.result_epoch_file is not None:
            Network.predict(self.model, self.preprocessed_dataset,
                            self.result_epoch_file.format(epoch))



from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.wrappers.scikit_learn import KerasClassifier

# Making sample_weight parameter explicit in KerasClassifier.fit method
# as expected by scikit_learn using a decorator technique
def decorate_kerasClassifier_fit(fit):
    def decorated_fit(self, x, y, sample_weight=None, **kwargs):
        return fit(self, x, y, sample_weight=sample_weight, **kwargs)

    return decorated_fit

KerasClassifier.fit = decorate_kerasClassifier_fit(KerasClassifier.fit)

# Evaluation - TODO: wrap model.fit with an evaluate method

class ModelBuilder:
    def __init__(self, preprocessed_dataset, vocabulary, word_embeddings_opt):
        self.preprocessed_dataset=preprocessed_dataset # TODO: to be changed to TwitterDataset
        self.vocabulary=vocabulary
        self.word_embeddings_opt=word_embeddings_opt
        self._created_models = []

    def register_ensemble(self, ensemble):
        self.ensemble = ensemble

    def __call__(self):
        # This code is copied from the Network.create_model method

        model = Sequential()

        # Create embedding layer
        word_embeddings_opt_param = {"initializer": "word2vec", "dim": 400, "trainable": False, "corpus_name": None}
        word_embeddings_opt_param.update(self.word_embeddings_opt)
        if word_embeddings_opt_param["initializer"] in Network.word_embedding_models:
            word_embeddings = Network.word_embedding_models[word_embeddings_opt_param["initializer"]](
                self.vocabulary, self.preprocessed_dataset,
                word_embeddings_opt_param["dim"], word_embeddings_opt_param["corpus_name"]) # TODO: Replace explicit dict access by **word_embeddings_opt
            embedding_layer = Embedding(self.vocabulary.word_count,
                                        word_embeddings_opt_param["dim"],
                                        weights=[word_embeddings.embedding_matrix],
                                        input_length=self.preprocessed_dataset.max_tweet_length,
                                        trainable=self.word_embeddings_opt_param["trainable"])

        else:
            embedding_layer = Embedding(self.vocabulary.word_count,
                                        word_embeddings_opt_param["dim"],
                                        input_length=self.preprocessed_dataset.max_tweet_length,
                                        trainable=self.word_embeddings_opt_param["trainable"])

        print("Created Embedding layer - Word count %d, dimensions %d, max tweet length %d" %
              (self.vocabulary.word_count, word_embeddings_opt_param["dim"], self.preprocessed_dataset.max_tweet_length))

        model.add(embedding_layer)
        model.add(LSTM(200))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print("Compiled model...")

        print(model.summary())

        # Decorate model.fit with writer of samples ranked by weights. This decorator may potentially also be used to
        # add preprocessing to the model (to use raw tweets as x)
        def decorate_kerasSequentialfit(receiver, fit):
            def wrapped_fit(x, y, batch_size=32, epochs=10, verbose=1, callbacks=None,
                            validation_split=0., validation_data=None, shuffle=True,
                            class_weight=None, sample_weight=None, initial_epoch=0, **kwargs):
                #TODO: next step is to generate vocabulary and preprocessor within keras model only from TwitterDataset
                #      based on sample_weights

                if sample_weight is not None:
                    # TODO: get a reference to preprocessed_dataset.shuffled_original_training_tweets

                    ranked_weights = [(i[1], i[0]) for i in enumerate(sample_weight)]
                    ranked_weights.sort()
                    # with open(training_samples_sorted_by_weight.format(phase), 'a+') as tssbwf:
                    print("\n***** Training samples sorted by weight *****\n")
                    for weight, i in ranked_weights:
                        print("\t{} :\t({})\t{}".format(weight, y[i], # TODO: Print original unpreprocessed strings
                                                        ' '.join([self.vocabulary.id_to_word[id] for id in x[i]]) ) ) # FIXME: expect self.vocabulary to change to receiver.vocabulary

                return fit(receiver, x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
                           validation_split=validation_split, validation_data=validation_data, shuffle=shuffle,
                           class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch, **kwargs)

            return wrapped_fit

        model.fit = decorate_kerasSequentialfit(model, Sequential.fit)

        model.ensemble = self.ensemble

        self._created_models.append(model)

        return model


# TODO: Make this class a single module as it maintains no internal state (apart from a list of word_embedding_models)
class Network:
    word_embedding_models =  { 'word2vec' : Word2VecEmbeddings,
                               'glove'    : GloVeEmbeddings}

    # TODO: Move this outside of this class to make this only dealing with Keras models (as opposed to Scikit Learn modles)
    @classmethod
    def create_model(cls,
                     preprocessed_dataset,
                     vocabulary,
                     word_embeddings_opt={},
                     model_builder=None):

        assert model_builder is not None

        # Create embedding layer
        word_embeddings_opt_param = {"initializer": "word2vec", "dim": 400, "trainable": False, "corpus_name": None}
        word_embeddings_opt_param.update(word_embeddings_opt)
        if word_embeddings_opt_param["initializer"] in Network.word_embedding_models:
            word_embeddings = Network.word_embedding_models[word_embeddings_opt_param["initializer"]](
                vocabulary, preprocessed_dataset,
                word_embeddings_opt_param["dim"], word_embeddings_opt_param["corpus_name"]) # TODO: Replace explicit dict access by **word_embeddings_opt
            embedding_layer = Embedding(vocabulary.word_count,
                                        word_embeddings_opt_param["dim"],
                                        weights=[word_embeddings.embedding_matrix],
                                        input_length=preprocessed_dataset.max_tweet_length,
                                        trainable=word_embeddings_opt_param["trainable"])

        else:
            embedding_layer = Embedding(vocabulary.word_count,
                                        word_embeddings_opt_param["dim"],
                                        input_length=preprocessed_dataset.max_tweet_length,
                                        trainable=word_embeddings_opt_param["trainable"])

        print("Created Embedding layer - Word count %d, dimensions %d, max tweet length %d" %
              (vocabulary.word_count, word_embeddings_opt_param["dim"], preprocessed_dataset.max_tweet_length))

        model=model_builder.get_model(embedding_layer)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print("Compiled model...")

        print(model.summary())

        return model


    @classmethod
    def create_adaboost_model(cls,
                              preprocessed_dataset,
                              vocabulary,
                              word_embeddings_opt={}):

        model_builder= ModelBuilder(preprocessed_dataset=preprocessed_dataset,
                                    vocabulary=vocabulary,
                                    word_embeddings_opt=word_embeddings_opt)

        #evaluater=ModelEvaluater(model, preprocessed_dataset, model, x_val, y_val, result_epoch_file=None) # problem: model, x_val, y_val not accessible at this time

        sklearn_model = KerasClassifier(build_fn=model_builder, epochs=3,
                                        batch_size=64, verbose=1 #, callbacks=[evaluater]
                                        )

        adaboost_model = AdaBoostClassifier(sklearn_model,
                                            algorithm="SAMME.R",
                                            n_estimators=3)

        # Store reference to AdaBoost instance in keras models to access AdaBoost internals at model fit time
        model_builder.register_ensemble(adaboost_model)

        return adaboost_model


    @classmethod
    def train_sklearn(cls,model, preprocessed_dataset,
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
        callbacks=[evaluater]

        if not config.test_run: # TODO: make callbacks accessible from config
            predicter=ModelPredicter(model, preprocessed_dataset, model_save_path, result_epoch_file)
            callbacks.append(predicter)

        model.fit(x_train, y_train, callbacks=callbacks, **training_opt_param)

        if model_save_path is not None:
            save_model(model, model_save_path)


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
        callbacks=[evaluater]

        if not config.test_run: # TODO: make callbacks accessible from config
            predicter=ModelPredicter(model, preprocessed_dataset, model_save_path, result_epoch_file)
            callbacks.append(predicter)

        model.fit(x_train, y_train, callbacks=callbacks, **training_opt_param)

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