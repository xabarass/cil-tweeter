import logging

import config

from TwitterDataset import TwitterDataSet

from Vocabulary import read_vocabulary_from_file, RegularizingPreprocessor, LexicalPreprocessor

from NeuralNetwork import Network

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def keras_model():
    print("Starting...")

    print("Loading tweets...")
    twitter_dataset = TwitterDataSet(positive_tweets=config.positive_tweets,
                                     negative_tweets=config.negative_tweets,
                                     test_data=config.test_data)

    print("Creating vocabulary...")
    word_to_occurrence_full = read_vocabulary_from_file(**config.vocab_path_opt)

    preprocessor = RegularizingPreprocessor(word_to_occurrence_full,**config.preprocessor_opt)
    #preprocessor = LexicalPreprocessor(word_to_occurrence_full,**config.preprocessor_opt)

    print("Creating training data set...")
    preprocessed_dataset = twitter_dataset.create_preprocessed_dataset(preprocessor, config.validation_split_ratio)

    print("Create model...")
    model = Network.create_model(
                          preprocessed_dataset=preprocessed_dataset,
                          preprocessor=preprocessor,
                          word_embeddings_opt=config.word_embeddings_opt,
                          model_builder=config.model_builder)

    print("Train model...")
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


if __name__ == '__main__':
    keras_model()