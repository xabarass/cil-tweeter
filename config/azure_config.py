import getpass
import Models

azure_config = None
local_config = None

user_name = getpass.getuser()

if user_name in {"nforster"}:
    azure_config = True
else:
    raise

# TBD: prepend output directory to output files
#def output_path_prefix(file_name):
#    file_path = "runs"

if azure_config:
    # Test run parameters
    test_run = False
else:
    raise

# Data set file paths

if azure_config:
    positive_tweets='./twitter-datasets/train_pos_full.txt'
    negative_tweets='./twitter-datasets/train_neg_full.txt'
    vocab_path='./twitter-datasets/vocab_full.txt'
else:
    raise

test_vocab_path='./twitter-datasets/test_vocab.txt'
test_data='./twitter-datasets/cleared_test_data.txt'

# Dataset parameters (size of validation data set)
if azure_config:
    validation_split_ratio=0.999
    test_run_data_ratio=1
else:
    raise

### Preprocessing options (token transformation/vocabulary generation)
vocab_path_opt = { "vocab_path": vocab_path  }

min_word_occurrence = 4
def final_vocabulary_filter(word, occurrence):
    return (len(word) > 3 and occurrence > min_word_occurrence) or \
           (len(word) == 3 and occurrence >= 1.25*min_word_occurrence) or \
           (len(word) == 2 and occurrence >= 10*min_word_occurrence)  or \
           (len(word) == 1 and occurrence >=50*min_word_occurrence)
final_vocabulary_filter.min_word_occurrence = min_word_occurrence # This is used by the Vocabulary and WordEmbedding classes

def preprocessor_vocabulary_filter(word, occurrence):
    return (len(word) > 3 and occurrence > min_word_occurrence) or \
           (len(word) == 3 and occurrence >= 3*min_word_occurrence) or \
           (len(word) == 2 and occurrence >= 100*min_word_occurrence) or \
           (len(word) == 1 and occurrence >= 1000*min_word_occurrence)

preprocessor_opt = { "remove_unknown_words": True,
                     "final_vocabulary_filter": final_vocabulary_filter,
                     "preprocessor_vocabulary_filter": preprocessor_vocabulary_filter}

### ML model options

# Embedding layer parameters
word_embeddings_opt = {"initializer": "word2vec",
                       "dim": 400,
                       "trainable": False,
                       "corpus_name": "full.emb"}

# Neural network parameter
model_builder=Models.SingleLSTM({"lstm_units":250})

# Training parameters
training_opt = {"epochs":3,
                "batch_size":64 }

# Results output parameters
result_file='results/result.csv'
misclassified_samples_file = 'misclassified_samples/misclassified_{}_samples'

# Load model parameters
model_save_path = "model"

