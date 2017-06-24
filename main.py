import datetime
import time
import logging

from NeuralNetwork import Network
import config
from TwitterDataset import TwitterDataSet, DefaultVocabularyTransformer, DefaultPreprocessor

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("Starting...")

data_set = TwitterDataSet(positive_tweets=config.positive_tweets,
                          negative_tweets=config.negative_tweets,
                          test_data=config.test_data,
                          vocab_path=config.vocab_path,
                          test_vocab_path=config.test_vocab_path)

preprocessor = DefaultPreprocessor(min_word_occurrence=4,
                                   remove_unknown_words=config.remove_unknown_words)
vocabulary_transformer = DefaultVocabularyTransformer(preprocessor)

vocabulary = data_set.create_vocabulary(vocabulary_transformer)

preprocessed_dataset = data_set.create_preprocessed_dataset(vocabulary, config.validation_split_ratio)

timestamp = str(int(time.time()))
result_file = ('_' + timestamp + '.').join( config.result_file.split('.') )
result_epoch_file = ('-e{}_' + timestamp + '.').join( config.result_file.split('.') )

trainModel=Network(config.word_embedding_dim)

trainModel.train(data_set,
                 config.generate_word_embeddings, config.embedding_corpus_name,
                 result_epoch_file=result_epoch_file,
                 misclassified_samples_file=config.misclassified_samples_file)
print("\tWriting to: {}".format(result_file))
trainModel.predict(data_set, result_file)
