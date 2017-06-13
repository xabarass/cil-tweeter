import logging

from NeuralNetwork import Network
import config
from data_set import TwitterDataSet

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("Starting...")

dataSet=TwitterDataSet(True,
                       positive_tweets=config.positive_tweets,
                       negative_tweets=config.negative_tweets,
                       test_data=config.test_data,
                       vocab_path=config.vocab_path,
                       remove_unknown_words=config.remove_unknown_words)

trainModel=Network(config.word_embedding_dim)
trainModel.train(dataSet, config.validation_split_ratio,
                 config.generate_word_embeddings, config.embedding_corpus_name)
# trainModel.load_model(config.model_json,config.model_h5)
trainModel.predict(dataSet, config.result_file)
