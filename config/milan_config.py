import getpass

azure_config = None
local_config = None

user_name = getpass.getuser()

if user_name in {"pmilan"}:
    azure_config = True
elif user_name in {"milan"}:
    local_config = True

# Data set file paths

if azure_config:
    positive_tweets='./twitter-datasets/train_pos_full.txt'
    negative_tweets='./twitter-datasets/train_neg_full.txt'
    vocab_path='./twitter-datasets/vocab_full.txt'
elif local_config:
    positive_tweets='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/train_pos.txt'
    negative_tweets='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/train_neg.txt'
    vocab_path='/home/milan/fax/computational_intelligence_lab/project/twitter-datasets/vocab.txt'
else:
    raise

test_vocab_path='./twitter-datasets/test_vocab.txt'
test_data='./twitter-datasets/cleared_test_data.txt'

# Dataset parameters (size of validation data set)
if azure_config:
    validation_split_ratio=0.99
elif local_config:
    validation_split_ratio = 0.5
else:
    raise

if azure_config:
    # Test run parameters
    test_run = False
    test_run_data_ratio=1
elif local_config:
    # Test run parameters
    test_run = False
    test_run_data_ratio=0.5
else:
    raise

# Vocabulary generation
preprocessor_opt = { "remove_unknown_words": True}

# TODO: Filter some of the very short and relatively rare words here <5-10 occurrences for length 3, <15-30 for length 2
def vocabulary_filter_factory(min_word_occurrence):
    def vocabulary_filter(word, occurrence):
        return occurrence >= min_word_occurrence
    vocabulary_filter.min_word_occurrence = min_word_occurrence # This is used by the Vocabulary and WordEmbedding classes
    return vocabulary_filter

vocabulary_transformer_opt = { "vocabulary_filter": vocabulary_filter_factory(4) }

# Embedding layer parameters
word_embeddings_opt = {"initializer": "word2vec",
                       "dim": 400,
                       "trainable": False,
                       "corpus_name": "full.emb"}

# Training parameters
training_opt = {"epochs":4,
                "batch_size":64 }

# Results output parameters
result_file='results/result.csv'
misclassified_samples_file = 'misclassified_samples/misclassified_{}_samples'

# Load model parameters
model_save_path = "model"

email="milanpandurov@gmail.com"