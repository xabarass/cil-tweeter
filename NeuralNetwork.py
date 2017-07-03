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
from keras.callbacks import TensorBoard

import Models
from Emailer import Emailer
import config

# import our own modules
from WordEmbeddings import Word2VecEmbeddings, GloVeEmbeddings
from KerasUtils import save_model
from TwitterDataset import PreprocessedDataset

############ Single neural network models ############

# Callbacks for logging during training
class ModelEvaluater(Callback):
    def __init__(self, model, x_val, y_val, verbosity=1, sample_weight=None):
        super(Callback, self).__init__()
        self.model=model
        self.x_val=x_val
        self.y_val=y_val
        self.verbosity=verbosity
        self.sample_weight=sample_weight
        if hasattr(config,'email'):
            self.emailer=Emailer('Training network update', config.email)
        else:
            self.emailer=None

    def on_epoch_end(self, epoch, logs=None):
        print("\nEvaluating epoch...")
        scores = self.model.evaluate(self.x_val, self.y_val, verbose=self.verbosity, sample_weight=self.sample_weight)
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


class Network:
    word_embedding_models =  { 'word2vec' : Word2VecEmbeddings,
                               'glove'    : GloVeEmbeddings}

    @classmethod
    def create_embedding_layer(cls,preprocessed_dataset,**word_embeddings_opt):
        preprocessor = preprocessed_dataset.preprocessor

        # Create embedding layer
        word_embeddings_opt_param = {"initializer": "word2vec",
                                     "dim": 400,
                                     "trainable": False,
                                     "corpus_name": None}
        word_embeddings_opt_param.update(word_embeddings_opt)
        if word_embeddings_opt_param["initializer"] in Network.word_embedding_models:
            word_embeddings = Network.word_embedding_models[word_embeddings_opt_param["initializer"]](
                preprocessor=preprocessor,
                preprocessed_tweets=preprocessed_dataset.all_preprocessed_tweets_weighted(),
                word_embedding_dimensions=word_embeddings_opt_param["dim"],
                embedding_corpus_name=word_embeddings_opt_param["corpus_name"])
            embedding_layer = Embedding(input_dim=preprocessor.vocabulary.word_count,
                                        output_dim=word_embeddings_opt_param["dim"],
                                        weights=[word_embeddings.embedding_matrix],
                                        input_length=preprocessed_dataset.max_tweet_length,
                                        trainable=word_embeddings_opt_param["trainable"])

        else:
            embedding_layer = Embedding(input_dim=preprocessor.vocabulary.word_count,
                                        output_dim=word_embeddings_opt_param["dim"],
                                        input_length=preprocessed_dataset.max_tweet_length,
                                        trainable=word_embeddings_opt_param["trainable"])

        print("Created Embedding layer - Word count %d, dimensions %d, max tweet length %d" %
              (preprocessor.vocabulary.word_count,
               word_embeddings_opt_param["dim"],
               preprocessed_dataset.max_tweet_length))
        return embedding_layer


    @classmethod
    def create_model(cls,
                     preprocessed_dataset,
                     word_embeddings_opt={},
                     model_builder=None):
        assert model_builder is not None

        embedding_layer = Network.create_embedding_layer(preprocessed_dataset, **word_embeddings_opt)

        model=model_builder.get_model(embedding_layer)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print("Compiled model...")

        print(model.summary())

        return model


    @classmethod
    def train(cls,model,
                  preprocessed_dataset,
                  training_opt={},
                  model_save_path=None,
                  result_epoch_file=None):

        training_opt_param = {"epochs":4, "batch_size":64}
        training_opt_param.update(training_opt)

        # Create training data
        (x_train, y_train, x_orig_train), (x_val, y_val, x_orig_val) = \
            preprocessed_dataset.shuffle_and_split_padded(model.input_names)

        evaluater=ModelEvaluater(model, x_val, y_val)
        callbacks=[evaluater]

        if not config.test_run: # TODO: make callbacks accessible from config
            predicter=ModelPredicter(model, preprocessed_dataset, model_save_path, result_epoch_file)
            callbacks.append(predicter)

        tensorBoard=TensorBoard(log_dir='./TensorBoard', histogram_freq=0,
                    write_graph=True, write_images=True)

        callbacks.append(tensorBoard)

        model.fit(x_train, y_train, callbacks=callbacks, **training_opt_param)

        if model_save_path is not None:
            save_model(model, model_save_path)

    @classmethod
    def output_misclassified_samples(cls,
                                     model, preprocessed_dataset, preprocessor,
                                     misclassified_samples_file=None):

        def evaluate_misclassified_samples(x, y, x_orig, phase):
            misclassified_samples = []

            x_padded = preprocessed_dataset.pad_tweets(x)
            pred_y = model.predict(x_padded, batch_size=64).reshape([-1])

            for i in range(pred_y.shape[0]):
                if ((pred_y[i] > 0.5) and (y[i] == 0)) or \
                   ((pred_y[i] <= 0.5) and (y[i] == 1)):
                    misclassified_samples.append(
                        ( 2*(pred_y[i]-0.5)*2*(y[i]-0.5), 2*(y[i]-0.5),
                        x_orig[i], ' '.join([preprocessor.vocabulary.id_to_word[id] for id in
                                             (x[i] if not isinstance(x,dict) else x['forward_input'][i] )]) ) )

            misclassified_samples.sort()

            with open(misclassified_samples_file.format(phase), 'a+') as mc_s_f:
                mc_s_f.write("\n***** Misclassified {} samples *****\n".format(phase))
                for sample in misclassified_samples:
                    mc_s_f.write( "\t{} :\t({})\n\t\t\t{}\n\t\t\t{}\n".format(
                                              sample[0], sample[1], sample[2], sample[3]) )

        print("Outputting misclassified samples...")
        (x_train, y_train, x_orig_train), (x_val, y_val, x_orig_val) = \
            preprocessed_dataset.shuffle_and_split(model.input_names)
        evaluate_misclassified_samples(x_val,  y_val, x_orig_val, "validation")
        evaluate_misclassified_samples(x_train, y_train, x_orig_train,"training")

    @classmethod
    def predict(cls, model, preprocessed_dataset, prediction_file):
        if not model:
            raise Exception("You need to train or load a pretrained model in order to predict")

        x_test = preprocessed_dataset.test_tweets_padded(model.input_names)
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


############ Boosted models ############


from sklearn.ensemble import AdaBoostClassifier
from keras.wrappers.scikit_learn import KerasClassifier

# Making sample_weight parameter explicit in KerasClassifier.fit method as expected by scikit_learn using a decorator
def decorate_kerasClassifier_fit(fit):
    def decorated_fit(self, x, y, sample_weight=None, **kwargs):
        history = fit(self, x, y, sample_weight=sample_weight, **kwargs)

        y_predict = self.predict(x)
        estimator_error = np.mean(
            np.average(y_predict != y, weights=sample_weight, axis=0))
        print("\n[KerasClassifier] Weighted training error: %.2f%%" % (estimator_error*100))
        return history
    return decorated_fit
KerasClassifier.fit = decorate_kerasClassifier_fit(KerasClassifier.fit)

def decorate_kerasClassifier_predict(predict):
    def decorated_predict(self, x, **kwargs):
        return np.reshape(predict(self, x,**kwargs), (-1,))
    return decorated_predict
KerasClassifier.predict = decorate_kerasClassifier_predict(KerasClassifier.predict)


# Sci-kit learn Adaboost derived class that stores current boosing iterate
class BoostingState:
    """Scikit learn AdaBoost iterate wrapper"""
    def __init__(self, iboost, X, sample_weight):
        self.iboost=iboost
        self.X=X
        self.sample_weight=sample_weight

# Log current state of boosting iteration
class AdaptiveAdaBoostClassifier(AdaBoostClassifier):
    """AdaBoost with exposed current boosting iterate"""
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Store current boosting iterate, then call base class implementation of this method."""
        print("\n[AdaptiveAdaBoostClassifier] Called _boost (%d-th iteration), logging boosting state..." % (iboost+1))
        self._boosting_state = BoostingState(iboost,X,sample_weight)
        return super(AdaptiveAdaBoostClassifier,self)._boost(iboost, X, y, sample_weight, random_state)


class AdaptiveKerasModelBuilder:
    """Keras model factory for vocabulary-adaptive (preprocessor-adaptive) AdaBoost ensemble with token-weighting for
       vocabulary generation and sample-weighting for word-embedding calculation.
       To be used with the KerasClassifier Scikit learn wrapper for keras Sequential models."""
    def __init__(self, twitter_dataset,
                       trivially_preprocessed_dataset,
                       preprocessor_factory,
                       word_embeddings_opt):
        self.twitter_dataset=twitter_dataset
        self.trivially_preprocessed_dataset=trivially_preprocessed_dataset
        self.preprocessor_factory=preprocessor_factory
        self.word_embeddings_opt=word_embeddings_opt
        self._created_models = []

    @classmethod
    def register_adaboost(cls,adaboost):
        cls._adaboost = adaboost

    def __call__(self):
        boosting_state = self._adaboost._boosting_state

        trivial_preprocessor = self.trivially_preprocessed_dataset.preprocessor
        unrenormalized_sample_weight_sum = np.sum(boosting_state.sample_weight)
        renormalized_sample_weights = [w / unrenormalized_sample_weight_sum * boosting_state.sample_weight.shape[0] # sum of all tweet's weight should be constant/each training tweet has a mean of sample weight 1
                                       for w in boosting_state.sample_weight]

        # Create preprocessor
        word_to_occurrence_full = {}

        print("[AdaptiveKerasModelBuilder] Boosting iteration %d: Creating new Keras model..." % (boosting_state.iboost+1) )
        trivially_preprocessed_tweets  = [trivial_preprocessor.map_id_seq_to_tweet(list(id_seq)) for id_seq in boosting_state.X]
        for tweet, weight in zip(trivially_preprocessed_tweets, renormalized_sample_weights):
            for word in tweet:
                if word in word_to_occurrence_full:
                    word_to_occurrence_full[word] += weight
                else:
                    word_to_occurrence_full[word] = weight

        #import pdb; pdb.set_trace()

        # introduce test data set to vocabulary # TODO: allow higher weighting!!
        for tweet in self.trivially_preprocessed_dataset.preprocessed_test_tweets:
            for word in tweet:
                if word in word_to_occurrence_full:
                    word_to_occurrence_full[word] += 1
                else:
                    word_to_occurrence_full[word] = 1

        preprocessor = self.preprocessor_factory(word_to_occurrence_full)
        preprocessed_dataset = PreprocessedDataset(self.twitter_dataset,
                                                   preprocessor,
                                                   config.validation_split_ratio)

        # Create model
        model = AdaptiveSequential(translator=None)
        model.translator = Translator(output_preprocessor=preprocessor,
                                      input_preprocessor=trivial_preprocessor,
                                      output_preprocessed_dataset=preprocessed_dataset)

        emb_preprocessed_tweets = []
        emb_sample_weights = renormalized_sample_weights
        for id_seq in model.translator(boosting_state.X):
            emb_preprocessed_tweets.append([w for w in filter(lambda word: word != '<pad>',preprocessor.map_id_seq_to_tweet(list(id_seq)))])

        # introduce test data set to word embeddings # TODO: allow higher weighting!!
        for tweet in preprocessed_dataset.preprocessed_test_tweets:
            emb_preprocessed_tweets.append(tweet)
            emb_sample_weights.append(1)

        # print("Tweets for word embeddings:")
        # for i in range(5):
        #     print(str(emb_sample_weights[i]) + " : " + str(emb_preprocessed_tweets[i][-5:]))
        # print("................")
        # print("................")
        # for i in range(5,0,-1):
        #     print(str(emb_sample_weights[-i]) + " : " + str(emb_preprocessed_tweets[-i][-5:]))

        #import pdb; pdb.set_trace()

        assert ("corpus_name" not in self.word_embeddings_opt) or (self.word_embeddings_opt["corpus_name"] is None)
        embedding_layer = AdaptiveAdaBoostModel.create_embedding_layer(preprocessed_train_tweets=emb_preprocessed_tweets,
                                                               sample_weight=emb_sample_weights,
                                                               preprocessed_dataset=preprocessed_dataset,
                                                               **self.word_embeddings_opt)
        #model.add(translation_layer)

        #TODO: use Models module
        model.add(embedding_layer)
        # model.add(LSTM(200))
        # model.add(Dropout(0.5))
        # model.add(Dense(1, activation='sigmoid'))

        model.add(Convolution1D(200,
                                5,
                                padding='valid',
                                activation='relu'))
        model.add(MaxPooling1D())
        model.add(Convolution1D(100,
                                3,
                                padding='valid',
                                activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))


        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print("Compiled model...")

        print(model.summary())

        self._created_models.append( (model,preprocessor) )

        return model



# Keras model concatenated with preprocessing used in vocabulary-adaptive AdaBoost ensembles
class Translator:
    """Translates Scikit learn input samples (padded tweets as int-sequences) from a larger vocabulary obtained with
       input_preprocessor to a smaller vocabulary obtained by output_processor (output_preprocessed_dataset is obtained
       by generating a preprocessed TwitterDataset with that preprocessor) using preprocessor functions to map from
       int-sequences to token-sequences, preprocess them and then map again to int-sequences."""
    def __init__(self, output_preprocessor, input_preprocessor, output_preprocessed_dataset):
        self.output_preprocessor = output_preprocessor
        self.input_preprocessor = input_preprocessor
        self.output_preprocessed_dataset = output_preprocessed_dataset

    def __call__(self, x):
        #print("[Translator] Translating input data set...")
        x_translated = np.zeros(shape=(x.shape[0], self.output_preprocessed_dataset.max_tweet_length), dtype=int)
        for i, id_seq in enumerate(x):
            reconstructed_input_token_seq = self.input_preprocessor.map_id_seq_to_tweet(list(id_seq))
            reconstructed_input_tweet = ' '.join([word for word in filter(lambda w: w != '<pad>',reconstructed_input_token_seq)])
            preprocessed_output_token_seq = self.output_preprocessor.preprocess_tweet(reconstructed_input_tweet)
            output_token_seq = self.output_preprocessor.map_tweet_to_id_seq(preprocessed_output_token_seq)
            x_translated[i, :] = np.array(self.output_preprocessed_dataset.pad_tweets( [output_token_seq] )[0])
            # print("[Translator] translated:")
            # print(str(reconstructed_input_token_seq[:1]) + ' ... ' + str(reconstructed_input_token_seq[-10:]))
            # print(str(self.output_preprocessor.map_id_seq_to_tweet(list(x_translated[i,:]))[:1]) + ' ... ' + str(self.output_preprocessor.map_id_seq_to_tweet(list(x_translated[i,:]))[-10:]))
        return x_translated


class AdaptiveSequential(Sequential):
    """Sequential keras model that applies preprocessing in a first step using translation from larger to
       smaller vocabulary"""
    def __init__(self,translator=None):
        print("[AdaptiveSequential] Creating AdaptiveSequential Keras model...")
        super().__init__()
        self.translator = translator

    def fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0, **kwargs):

        # if sample_weight is not None:
        #     # TODO: get a reference to preprocessed_dataset.shuffled_original_training_tweets
        #
        #     ranked_weights = [(i[1], i[0]) for i in enumerate(sample_weight)]
        #     ranked_weights.sort()
        #     # with open(training_samples_sorted_by_weight.format(phase), 'a+') as tssbwf:
        #     print("\n***** Training samples sorted by weight *****\n")
        #     for weight, i in ranked_weights:
        #         print("\t{} :\t({})\t{}".format(weight, y[i], # TODO: Print original unpreprocessed strings
        #                                         ' '.join([self.preprocessor.vocabulary.id_to_word[id] for id in x[i]]) ) ) # FIXME: expect self.vocabulary to change to receiver.vocabulary

        evaluater=ModelEvaluater(self, x, y, verbosity=1, sample_weight=sample_weight)
        callbacks = [evaluater] if callbacks is None else (callbacks + [evaluater])
        # Note, the validation accuracy displayed here is actually the weighted training accurracy

        return super().fit(self.translator(x), y,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose,
                              callbacks=callbacks,
                              validation_split=validation_split,
                              validation_data=validation_data,
                              shuffle=shuffle,
                              class_weight=class_weight,
                              sample_weight=sample_weight,
                              initial_epoch=initial_epoch)


    def evaluate(self, x, y, batch_size=32, verbose=1,
                 sample_weight=None):
        return super().evaluate(self.translator(x), y,
                                   batch_size=batch_size,
                                   verbose=verbose,
                                   sample_weight=sample_weight)

    def predict(self, x, batch_size=32, verbose=0):
        return super().predict(self.translator(x),
                                              batch_size=batch_size,
                                              verbose=verbose)


    def predict_on_batch(self, x):
        return super().predict_on_batch(self.translator(x))

    def train_on_batch(self, x, y, class_weight=None,
                       sample_weight=None):
        return super().train_on_batch(self.translator(x), y,
                                         class_weight=class_weight,
                                         sample_weight=sample_weight)

    def test_on_batch(self, x, y,
                      sample_weight=None):
        return super().test_on_batch(self.translator(x), y,sample_weight=None)

    def fit_generator(self, generator,
                      steps_per_epoch,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_steps=None,
                      class_weight=None,
                      max_q_size=10,
                      workers=1,
                      pickle_safe=False,
                      initial_epoch=0):
        raise Exception("Generator interface not supported by this keras.models.Sequential wrapper")

    def evaluate_generator(self, generator, steps,
                           max_q_size=10, workers=1,
                           pickle_safe=False):
        raise Exception("Generator interface not supported by this keras.models.Sequential wrapper")

    def predict_generator(self, generator, steps,
                          max_q_size=10, workers=1,
                          pickle_safe=False, verbose=0):
        raise Exception("Generator interface not supported by this keras.models.Sequential wrapper")



class StaticKerasModelBuilder:
    """Keras model factory for static-preprocessor AdaBoost ensemble.
       To be used with the KerasClassifier Scikit learn wrapper for keras Sequential models."""
    def __init__(self, preprocessed_dataset,
                       word_embeddings_opt,
                       model_builder):
        self.preprocessed_dataset=preprocessed_dataset
        self.word_embeddings_opt=word_embeddings_opt
        self.model_builder=model_builder
        self._created_models = []

    @classmethod
    def register_adaboost(cls,adaboost):
        cls._adaboost = adaboost

    def __call__(self):
        boosting_state = self._adaboost._boosting_state
        # Use config.ensemble_model_builder to configure model generation
        model = Network.create_model(preprocessed_dataset=self.preprocessed_dataset,
                                     word_embeddings_opt=self.word_embeddings_opt,
                                     model_builder=self.model_builder)
        self._created_models.append( model )
        return model


class AdaptiveAdaBoostModel:
    @classmethod
    def create_embedding_layer(cls, preprocessed_train_tweets, sample_weight, preprocessed_dataset,**word_embeddings_opt):
        preprocessor = preprocessed_dataset.preprocessor

        # Create embedding layer
        word_embeddings_opt_param = {"initializer": "word2vec",
                                     "dim": 400,
                                     "trainable": False,
                                     "corpus_name": None}
        word_embeddings_opt_param.update(word_embeddings_opt)
        if word_embeddings_opt_param["initializer"] in Network.word_embedding_models:
            word_embeddings = Network.word_embedding_models[word_embeddings_opt_param["initializer"]](
                preprocessor=preprocessor,
                preprocessed_tweets=preprocessed_dataset.weighted_preprocessed_tweets(preprocessed_train_tweets=preprocessed_train_tweets,
                                                                                      sample_weight=sample_weight),
                word_embedding_dimensions=word_embeddings_opt_param["dim"],
                embedding_corpus_name=word_embeddings_opt_param["corpus_name"])
            embedding_layer = Embedding(input_dim=preprocessor.vocabulary.word_count,
                                        output_dim=word_embeddings_opt_param["dim"],
                                        weights=[word_embeddings.embedding_matrix],
                                        input_length=preprocessed_dataset.max_tweet_length,
                                        trainable=word_embeddings_opt_param["trainable"])

        else:
            embedding_layer = Embedding(input_dim=preprocessor.vocabulary.word_count,
                                        output_dim=word_embeddings_opt_param["dim"],
                                        input_length=preprocessed_dataset.max_tweet_length,
                                        trainable=word_embeddings_opt_param["trainable"])

        print("Created Embedding layer - Word count %d, dimensions %d, max tweet length %d" %
              (preprocessor.vocabulary.word_count,
               word_embeddings_opt_param["dim"],
               preprocessed_dataset.max_tweet_length))
        return embedding_layer

    @classmethod
    def create_model(cls,twitter_dataset,
                         trivially_preprocessed_dataset,
                         preprocessor_factory,
                         word_embeddings_opt={}, # result_epoch_file=None
                         training_opt={},
                         adaboost_opt={}):

        model_builder= AdaptiveKerasModelBuilder(
                           twitter_dataset=twitter_dataset,
                           trivially_preprocessed_dataset=trivially_preprocessed_dataset,
                           preprocessor_factory=preprocessor_factory,
                           word_embeddings_opt=word_embeddings_opt)

        training_opt_param = {"epochs":4, "batch_size":64}
        training_opt_param.update(training_opt)

        adaboost_opt_param = { "algorithm": "SAMME.R",
                               "n_estimators": 5,
                               "learning_rate": 1}
        adaboost_opt_param.update(adaboost_opt)

        # evaluater=ModelEvaluater(model, x_val, y_val)
        # callbacks=[evaluater] # TODO: using boosting state

        # if not config.test_run: # TODO: make callbacks accessible from config
        #     predicter=ModelPredicter(model, preprocessed_dataset, model_save_path, result_epoch_file)
        #     callbacks.append(predicter)

        sklearn_model = KerasClassifier(build_fn=model_builder,
                                        verbose=1, **training_opt_param #, callbacks=callbacks
                                        )

        adaboost_model = AdaptiveAdaBoostClassifier(sklearn_model,
                                                    **adaboost_opt_param)

        # Store reference to AdaBoost instance in keras models to access AdaBoost internals at model fit time
        model_builder.register_adaboost(adaboost_model)

        return adaboost_model

class StaticAdaBoostModel:
    @classmethod
    def create_model(cls,
                     preprocessed_dataset,
                     word_embeddings_opt={},
                     training_opt={},
                     adaboost_opt={},
                     model_builder=None):
        keras_model_factory = StaticKerasModelBuilder(
            preprocessed_dataset=preprocessed_dataset,
            word_embeddings_opt=word_embeddings_opt,
            model_builder=model_builder)

        training_opt_param = {"epochs": 4, "batch_size": 64}
        training_opt_param.update(training_opt)

        adaboost_opt_param = { "algorithm": "SAMME.R",
                               "n_estimators": 5,
                               "learning_rate": 1}
        adaboost_opt_param.update(adaboost_opt)


        sklearn_model = KerasClassifier(build_fn=keras_model_factory,
                                        verbose=1, **training_opt_param  # , callbacks=callbacks
                                        )

        adaboost_model = AdaptiveAdaBoostClassifier(sklearn_model,
                                                    **adaboost_opt_param)

        # Store reference to AdaBoost instance in keras models to access AdaBoost internals at model fit time
        keras_model_factory.register_adaboost(adaboost_model)

        return adaboost_model


class AdaBoostModel:
    @classmethod
    def train(cls,model,
                  preprocessed_dataset, # NOTE: this dataset is trivially preprocessed (LexicalPreprocessor only)
                  model_save_path=None):


        # Create training data
        (x_train, y_train, x_orig_train), (x_val, y_val, x_orig_val) = preprocessed_dataset.shuffle_and_split_padded()

        model.fit(x_train, y_train)

        print("***** Training summary *****")
        for iboost, (weight, error) in enumerate(zip(model.estimator_weights_,model.estimator_errors_)):
            print( "\t%d-th estimator: weighted training error = %.2f%%, estimator weight = %.8f" % (iboost, 100*error, weight) )

        print("***** Evaluation *****")
        for iboost, accuracy in enumerate(model.staged_score(x_val, y_val)):
            print( "\tAfter %d-th boosting iteration: accuracy = %.2f%%" % (iboost, 100*accuracy) )

        # TODO: save all models and weights to a json file (model.weights, etc. cf. doc)
        #if model_save_path is not None:
        #    save_model(model, model_save_path)

#    @classmethod
#    def output_misclassified_samples(cls,
#                                     model, preprocessed_dataset, preprocessor,
#                                     misclassified_samples_file=None):

    @classmethod
    def predict(cls, model, preprocessed_dataset, prediction_file):
        if not model:
            raise Exception("You need to train or load a pretrained model in order to predict")

        x_test = preprocessed_dataset.test_tweets_padded()
        predictions = model.predict(x_test)

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
