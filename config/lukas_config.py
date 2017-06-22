import getpass

azure_config = None
local_config = None

user_name = getpass.getuser()

if user_name in {"lukasd"}:
    azure_config = True
elif user_name in {"lukas"}:
    local_config = True

if azure_config:
    # Configuration
    positive_tweets='./twitter-datasets/train_pos_full.txt'
    negative_tweets='./twitter-datasets/train_neg_full.txt'
    vocab_path='./twitter-datasets/vocab_full.txt'
elif local_config:
    # Configuration
    positive_tweets='./twitter-datasets/train_pos.txt'
    negative_tweets='./twitter-datasets/train_neg.txt'
    vocab_path='./twitter-datasets/vocab.txt'
else:
    raise

test_data='./twitter-datasets/cleared_test_data.txt'
remove_unknown_words=True

# Further model parameters of main.py
word_embedding_dim=400
if azure_config:
    validation_split_ratio=0.99
elif local_config:
    validation_split_ratio = 0.01
else:
    raise
result_file='results/result.csv'

# Further training parameters
generate_word_embeddings=True
embedding_corpus_name='full.emb'

# Load model parameters
model_json = 'model.json'
model_h5 = 'model.h5'
