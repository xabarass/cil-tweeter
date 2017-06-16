# Configuration
positive_tweets='./twitter-datasets/train_pos_full.txt'
negative_tweets='./twitter-datasets/train_neg_full.txt'
test_data='./twitter-datasets/cleared_test_data.txt'
vocab_path='./twitter-datasets/vocab_full.txt'
remove_unknown_words=True

# Further model parameters of main.py
word_embedding_dim=300
validation_split_ratio=0.99
result_file='results/result.csv'

# Further training parameters
generate_word_embeddings=True
embedding_corpus_name='small.emb'

# Load model parameters
model_json = 'model.json'
model_h5 = 'model.h5'
