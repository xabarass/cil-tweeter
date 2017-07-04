import logging

import config

from TwitterDataset import TwitterDataSet

from Vocabulary import read_vocabulary_from_file, ouput_vocabulary_statistics, RegularizingPreprocessor, LexicalPreprocessor, StemmingPreprocessor, CharacterBasedPreprocessor

from NeuralNetwork import Network, AdaBoostModel, StaticAdaBoostModel, AdaptiveAdaBoostModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def keras_model():
    print("Starting keras_model...")

    print("Loading tweets...")
    twitter_dataset = TwitterDataSet(positive_tweets=config.positive_tweets,
                                     negative_tweets=config.negative_tweets,
                                     test_data=config.test_data)

    print("Creating vocabulary...")
    word_to_occurrence_full = read_vocabulary_from_file(**config.vocab_path_opt)

    # preprocessor = RegularizingPreprocessor(word_to_occurrence_full,**config.preprocessor_opt)
    preprocessor = LexicalPreprocessor(word_to_occurrence_full,**config.preprocessor_opt)
    preprocessor = StemmingPreprocessor(preprocessor,
                                       stemming_vocabulary_filter=config.preprocessor_opt['final_vocabulary_filter'],
                                       remove_unknown_words=config.preprocessor_opt['remove_unknown_words'])
    # preprocessor=CharacterBasedPreprocessor(word_to_occurrence_full)

    print("Preprocessing training data set...")
    preprocessed_dataset = twitter_dataset.create_preprocessed_dataset(preprocessor, config.validation_split_ratio)

    print("Create keras model...")
    model = Network.create_model(
                          preprocessed_dataset=preprocessed_dataset,
                          word_embeddings_opt=config.word_embeddings_opt,
                          model_builder=config.model_builder)

    print("Train keras model...")
    Network.train(model=model,
                  preprocessed_dataset=preprocessed_dataset,
                  training_opt=config.training_opt,
                  model_save_path=config.model_save_path,
                  result_epoch_file=config.result_epoch_file)

    print("Output misclassified samples...")
    Network.output_misclassified_samples(model=model,
                          preprocessed_dataset=preprocessed_dataset, preprocessor=preprocessor,
                          misclassified_samples_file=config.misclassified_samples_file)

    print("Output predictions...")
    print("\tWriting to: {}".format(config.result_file))
    Network.predict(model=model,
                    preprocessed_dataset=preprocessed_dataset,
                    prediction_file=config.result_file)

def static_adaboost_model():
    print("Starting static AdaBoost sklearn_model...")

    print("Loading tweets...")
    twitter_dataset = TwitterDataSet(positive_tweets=config.positive_tweets,
                                     negative_tweets=config.negative_tweets,
                                     test_data=config.test_data)

    print("Creating vocabulary...")
    word_to_occurrence_full = read_vocabulary_from_file(**config.vocab_path_opt)

    preprocessor = RegularizingPreprocessor(word_to_occurrence_full,**config.preprocessor_opt)
    #preprocessor = LexicalPreprocessor(word_to_occurrence_full,**config.preprocessor_opt)

    print("Preprocessing training data set...")
    preprocessed_dataset = twitter_dataset.create_preprocessed_dataset(preprocessor, config.validation_split_ratio)

    print("Create static sklearn AdaBoost model...")
    model = StaticAdaBoostModel.create_model(
                          preprocessed_dataset=preprocessed_dataset,
                          word_embeddings_opt=config.word_embeddings_opt,
                          training_opt=config.training_opt,
                          adaboost_opt=config.adaboost_opt,
                          model_builder=config.ensemble_model_builder)

    print("Train sklearn model...")
    AdaBoostModel.train(model=model,
                        preprocessed_dataset=preprocessed_dataset)

    # print("Output misclassified samples...")
    # Network.output_misclassified_samples(model=model,
    #                                      preprocessed_dataset=trivially_preprocessed_dataset,
    #                                      preprocessor=trivial_preprocessor,
    #                                      misclassified_samples_file=config.misclassified_samples_file)

    print("Output predictions...")
    print("\tWriting to: {}".format(config.result_file))
    AdaBoostModel.predict(model=model,
                    preprocessed_dataset=preprocessed_dataset,
                    prediction_file=config.result_file)



def adaptive_adaboost_model():
    print("Starting adaptive AdaBoost sklearn_model...")

    print("Loading tweets...")
    twitter_dataset = TwitterDataSet(positive_tweets=config.positive_tweets,
                                     negative_tweets=config.negative_tweets,
                                     test_data=config.test_data,
                                     deduplicate_train_tweets=False)

    print("Creating vocabulary...")
    word_to_occurrence_full = read_vocabulary_from_file(**config.vocab_path_opt)

    preprocessor_factory = lambda word_to_occurrence: RegularizingPreprocessor(word_to_occurrence,
                                                                               **config.preprocessor_opt)

    trivial_preprocessor = LexicalPreprocessor(word_to_occurrence_full=word_to_occurrence_full,
                                               final_vocabulary_filter=lambda word, occurrence: True,
                                               remove_unknown_words=False)

    print("Preprocessing training data set...")
    trivially_preprocessed_dataset = twitter_dataset.create_preprocessed_dataset(trivial_preprocessor, config.validation_split_ratio)

    print("Create sklearn model...")
    model = AdaptiveAdaBoostModel.create_model(
                          twitter_dataset=twitter_dataset,
                          trivially_preprocessed_dataset=trivially_preprocessed_dataset,
                          preprocessor_factory=preprocessor_factory,
                          word_embeddings_opt=config.word_embeddings_opt,
                          training_opt = config.training_opt,
                          adaboost_opt=config.adaboost_opt)

    print("Train sklearn model...")
    AdaBoostModel.train(model=model,
                  preprocessed_dataset=trivially_preprocessed_dataset,
                  model_save_path=config.model_save_path) #TODO: use boost-id to in result_epoch_file

    # print("Output misclassified samples...")
    # Network.output_misclassified_samples(model=model,
    #                                      preprocessed_dataset=trivially_preprocessed_dataset,
    #                                      preprocessor=trivial_preprocessor,
    #                                      misclassified_samples_file=config.misclassified_samples_file)

    print("Output predictions...")
    print("\tWriting to: {}".format(config.result_file))
    AdaBoostModel.predict(model=model,
                    preprocessed_dataset=trivially_preprocessed_dataset,
                    prediction_file=config.result_file)


def print_vocabulary_statistics():
    print("Creating vocabulary...")
    ouput_vocabulary_statistics(read_vocabulary_from_file(**config.vocab_path_opt))

if __name__ == '__main__':
    keras_model()

