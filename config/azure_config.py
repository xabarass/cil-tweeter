import getpass
import Models

azure_config = None
local_config = None

user_name = getpass.getuser()

if user_name in {"nforster"}:
    azure_config = True

# Data set file paths

if azure_config:
    positive_tweets='./datasets/train_pos_full.txt'
    negative_tweets='./datasets/train_neg_full.txt'
    vocab_path='./datasets/vocab_full.txt'
else:
    raise

test_vocab_path='./datasets/test_vocab.txt'
test_data='./datasets/cleared_test_data.txt'

# Dataset parameters (size of validation data set)
if azure_config:
    print("Running Azure config!")
    validation_split_ratio=0.99
else:
    raise

if azure_config:
    # Test run parameters
    test_run = False
    test_run_data_ratio=1
else:
    raise

# Vocabulary generation
preprocessor_opt = { "remove_unknown_words": True}

min_word_occurrence = 4
def vocabulary_filter(word, occurrence):
    return (len(word) > 3 and occurrence > min_word_occurrence) or \
           (len(word) == 3 and occurrence >= 1.25*min_word_occurrence) or \
           (len(word) == 2 and occurrence >= 10*min_word_occurrence)  or \
           (len(word) == 1 and occurrence >=50*min_word_occurrence)
vocabulary_filter.min_word_occurrence = min_word_occurrence # This is used by the Vocabulary and WordEmbedding classes

vocabulary_opt = { "vocabulary_filter": vocabulary_filter }

def hashtag_multiplier(word):
    return 20 if word.startswith('#') else 1
def vocabulary_generator_filter(word, occurrence):
    return (len(word) > 3  and occurrence > hashtag_multiplier(word)*min_word_occurrence) or \
           (len(word) == 3 and occurrence >= hashtag_multiplier(word)*3*min_word_occurrence) or \
           (len(word) == 2 and occurrence >= hashtag_multiplier(word)*100*min_word_occurrence) or \
           (len(word) == 1 and occurrence >= hashtag_multiplier(word)*1000*min_word_occurrence)

vocabulary_generator_opt = { "vocabulary_generator_filter": vocabulary_generator_filter }

# Embedding layer parameters
word_embeddings_opt = {"initializer": "word2vec",
                       "dim": 400,
                       "trainable": False,
                       "corpus_name": "full.emb"}

# Model parameter
model_builder=Models.DoubleConv()

# Training parameters
training_opt = {"epochs":3,
                "batch_size":64 }

# Results output parameters
result_file='results/result.csv'
misclassified_samples_file = 'misclassified_samples/misclassified_{}_samples'

# Load model parameters
model_save_path = "model"

email="milanpandurov@gmail.com"
