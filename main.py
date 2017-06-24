import datetime
import time
import logging

from NeuralNetwork import Network
import config
from TwitterDataset import TwitterDataSet, DefaultVocabularyTransformer, DefaultPreprocessor

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("Starting...")

print("Loading tweets...")
data_set = TwitterDataSet(positive_tweets=config.positive_tweets,
                          negative_tweets=config.negative_tweets,
                          test_data=config.test_data,
                          vocab_path=config.vocab_path,
                          test_vocab_path=config.test_vocab_path)

print("Creating vocabulary...")
preprocessor = DefaultPreprocessor(**config.preprocessor_opt)
vocabulary_transformer = DefaultVocabularyTransformer(preprocessor)

vocabulary = data_set.create_vocabulary(vocabulary_transformer)

print("Preprocessing data set...")
preprocessed_dataset = data_set.create_preprocessed_dataset(vocabulary, config.validation_split_ratio)

timestamp = str(int(time.time()))
result_file = ('_' + timestamp + '.').join( config.result_file.split('.') )
result_epoch_file = ('-e{}_' + timestamp + '.').join( config.result_file.split('.') )

print("Create model...")
model = Network.create_model(
                      preprocessed_dataset=preprocessed_dataset,
                      vocabulary=vocabulary,
                      word_embeddings_opt=config.word_embeddings_opt)


print("Train model...")
Network.train(model=model,
              preprocessed_dataset=preprocessed_dataset,
              training_opt=config.training_opt,
              model_save_path=config.model_save_path,
              result_epoch_file=result_epoch_file)

print("Output misclassified samples...")
Network.output_misclassified_samples(model=model,
                      preprocessed_dataset=preprocessed_dataset, vocabulary=vocabulary,
                      misclassified_samples_file=config.misclassified_samples_file)

print("Output predictions...")
print("\tWriting to: {}".format(result_file))
Network.predict(model=model,
                preprocessed_dataset=preprocessed_dataset,
                prediction_file=result_file)
