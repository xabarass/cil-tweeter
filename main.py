import logging
from data_set import TwitterDataSet
from NeuralNetwork import Network

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("Starting...")

dataSet=TwitterDataSet(True,
                       positive_tweets='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/train_pos.txt',
                       negative_tweets='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/train_neg.txt',
                       test_data='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/test_data.txt',
                       vocab_path='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/vocab.txt',
                       remove_unknown_words=True)

word_embedding_dim=200
validation_split_ratio=0.99
result_file='result.csv'

trainModel=Network(word_embedding_dim)
trainModel.train(dataSet, validation_split_ratio, generate_word_embeddings=True, embedding_corpus_name='small.emb')
# trainModel.predict(dataSet, result_file)
