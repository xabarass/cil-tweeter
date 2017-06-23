# Configuration data

# Twitter-Dataset constructor
positive_tweets='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/train_pos.txt'
negative_tweets='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/train_neg.txt'
test_data='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/test_data.txt'
vocab_path='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/vocab.txt'
remove_unknown_words=True

# Further model parameters of main.py
word_embedding_dim=400
validation_split_ratio=0.8
result_file='result.csv'

# Further training parameters
generate_word_embeddings=True
embedding_corpus_name='small.emb'

# Load model parameters
model_json = 'model.json'
model_h5 = 'model.h5'

# Test run parameters
test_run = True
test_run_data_ratio=0.01

