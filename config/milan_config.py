import getpass

azure_config = None
local_config = None

user_name = getpass.getuser()

if user_name in {"pmilan"}:
    azure_config = True
elif user_name in {"milan"}:
    local_config = True

if local_config:
    # Configuration
    positive_tweets='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/train_pos.txt'
    negative_tweets='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/train_neg.txt'
    vocab_path='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/vocab.txt'

    test_vocab_path = '/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/test_only_vocab.txt'
    test_data = '/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/cleared_test_data.txt'
elif azure_config:
    # Configuration
    positive_tweets='./twitter-datasets/train_pos.txt'
    negative_tweets='./twitter-datasets/train_neg.txt'
    vocab_path='./twitter-datasets/vocab.txt'
else:
    raise

remove_unknown_words=True

# Further model parameters of main.py
word_embedding_dim=400
if azure_config:
    validation_split_ratio=0.99
elif local_config:
    validation_split_ratio = 0.8
else:
    raise

result_file='results/result.csv'
misclassified_samples_file = 'misclassified_samples/misclassified_{}_samples'


# Further training parameters
generate_word_embeddings=True
embedding_corpus_name='full.emb'

# Load model parameters
model_json = 'model.json'
model_h5 = 'model.h5'

if azure_config:
    # Test run parameters
    test_run = False
    test_run_data_ratio=1
elif local_config:
    # Test run parameters
    test_run = True
    test_run_data_ratio=1
else:
    raise