import logging
from data_set import TwitterDataSet
from NeuralNetwork import Network

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("Starting...")

dataSet=TwitterDataSet(True,
                       positive_tweets='/tmp/twitter-datasets/train_pos.txt',
                       negative_tweets='/tmp/twitter-datasets/train_neg.txt',
                       test_data='/tmp/twitter-datasets/test_data.txt',
                       vocab_path='/tmp/twitter-datasets/vocab.txt')

trainModel=Network(300)
trainModel.train(dataSet, 0.8)
