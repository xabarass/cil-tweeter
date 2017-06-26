import pickle
import logging
import os
import numpy as np
from tqdm import tqdm
from TextRegularizer import TextRegularizer


class Vocabulary:
    def __init__(self, vocab_transformer, vocabulary_filter, vocab_path, test_vocab_path=None):
        """Load and compute vocabulary from output of twitter-datasets/build_vocab.sh"""
        if vocab_transformer is None:
             raise Exception("Not a valid vocabulary transformer supplied: %s" % repr(vocab_transformer))
        if not os.path.isfile(vocab_path):
            raise Exception("Not a valid file: %s" % vocab_path)

        # Initialize vocabulary state
        self.word_to_id = {}
        self.word_to_occurrence = {}
        self.vocabulary_filter=vocabulary_filter

        # Refer to vocab to enable online tweet preprocessing from within the preprocessor
        self.preprocessor = vocab_transformer.preprocessor
        vocab_transformer.preprocessor.register_vocabulary(self)

        if hasattr(vocabulary_filter,"min_word_occurrence"):
            self.min_word_occurrence = vocabulary_filter.min_word_occurrence
        else:
            raise

        print("Loading vocabulary..")

        # Load test vocabulary
        if test_vocab_path is not None:
            test_vocab = {}
            with open(test_vocab_path, 'r') as test_vocab_file:
                for line in test_vocab_file:
                    tokens = line.split(' ')
                    occurrence = int(tokens[-2])
                    word = tokens[-1]
                    assert word not in test_vocab
                    test_vocab[word] = occurrence

        # Load merged train and test vocabulary
        word_to_occurrence={}
        with open(vocab_path, 'r') as vocab:
            for line in vocab:
                tokens = line.split(' ')
                occurrence = int(tokens[-2])
                word = tokens[-1].rstrip()
                if word not in word_to_occurrence:
                    word_to_occurrence[word] = occurrence
                else:
                    word_to_occurrence[word] += occurrence

        # vocab_transformer defines vocabulary by injecting extra symbols and filtering the generated vocabulary
        self.word_to_id, self.word_to_occurrence = vocab_transformer(word_to_occurrence)

        # build reverse lookup dictionary
        self.id_to_word = {}
        for word in self.word_to_id:
            self.id_to_word[self.word_to_id[word]] = word

        print("Vocabulary of model has {} words".format(len(self.word_to_id)))

        # print("*********** Computing short word sorted by frequencies ***********")
        # short_words = {1: [], 2: [], 3: []}
        # for word, occurrence in self.word_to_occurrence.items():
        #     if len(word) < 4 and occurrence > 10:
        #         short_words[len(word)].append((-occurrence, word))
        #
        # for size, words in short_words.items():
        #     words.sort()
        #     print("Words of length %d" % size)
        #     for occurrence, word in words:
        #         print("\t %d : %s" % (-occurrence, word))

    @property
    def word_count(self):
        return len(self.word_to_id)

    def preprocess_and_map_tweet_to_id_seq(self, tweet):
        """Preprocess tweet with lexical/stemming/filtering phase and replace every token by vocabulary integer id"""
        assert isinstance(tweet,str)
        token_seq = self.preprocessor.preprocess_tweet(tweet)
        return self.map_tweet_to_id_seq(token_seq)

    def map_tweet_to_id_seq(self, tweet_token_seq):
        """Replace every token by vocabulary integer id"""
        assert isinstance(tweet_token_seq, list)
        return [self.word_to_id[word] for word in tweet_token_seq]

    def map_id_seq_to_tweet(self, tweet_id_seq):
        """Replace every vocabulary integer id by token"""
        assert isinstance(tweet_id_seq, list)
        return [self.id_to_word[id] for id in tweet_id_seq]


class DefaultVocabularyTransformer:
    """Function object that creates word_to_id vocabulary dictionary from word_to_occurrence statistics of corpus"""
    def __init__(self, preprocessor, vocabulary_transformer_filter):
        self.preprocessor = preprocessor
        self.vocabulary_filter = vocabulary_transformer_filter

    def __call__(self, word_to_occurrence_full):
        self.vocab_preprocessor = self.preprocessor.clone_shallow() # create a deep copy of the preprocessor to be used for vocabulary generation

        class VocabularyProxy:
            """Creates a filtered copy of currently preprocessed words to serve as a temporary vocabulary"""
            def __init__(self, unfiltered_word_to_occurrence, vocabulary_filter):
                self.word_to_occurrence = {}
                for word, occurrence in unfiltered_word_to_occurrence.items():
                    if vocabulary_filter(word,occurrence):
                        self.word_to_occurrence[word] = unfiltered_word_to_occurrence[word]

        # Two passes of vocabulary creation
        # First collect words by frequency, in a second phase create additional words
        word_to_preprocessed_words = {}
        preprocessed_word_to_occurrence = {}
        self.vocab_preprocessor.register_vocabulary(
            VocabularyProxy(preprocessed_word_to_occurrence, self.vocabulary_filter) )

        for word in tqdm(word_to_occurrence_full, desc="[VocabularyTransformer] - initial_pass"):
            preprocessed_words = self.vocab_preprocessor.initial_pass_vocab(word) # preprocessor returns a list of words that word parameter gets preprocessed into
            assert isinstance(preprocessed_words, list)
            word_to_preprocessed_words[word] = preprocessed_words

        for word, preprocessed_words in word_to_preprocessed_words.items():
            for preprocessed_word in preprocessed_words:
                if preprocessed_word not in preprocessed_word_to_occurrence:
                    preprocessed_word_to_occurrence[preprocessed_word] = word_to_occurrence_full[word]
                else:
                    preprocessed_word_to_occurrence[preprocessed_word] += word_to_occurrence_full[word]

        self.vocab_preprocessor.register_vocabulary(
            VocabularyProxy(preprocessed_word_to_occurrence, self.vocabulary_filter) )

        extra_pass_count = 0
        while True:
            replaced_preprocessed_words = {}
            #print("\t[DefaultVocabularyTransformer]\t - %d-th extra pass" % (extra_pass_count+1))
            for word in tqdm(word_to_occurrence_full, desc="[VocabularyTransformer] - extra_pass %d" % (extra_pass_count + 1) ):
                preprocessed_words = self.vocab_preprocessor.extra_pass_vocab(word) # preprocessor returns a list of words that word parameter gets preprocessed into
                assert isinstance(preprocessed_words, list)
                if preprocessed_words != word_to_preprocessed_words[word]:
                    #print("\t '{}':\n\t\t[{}]\t --> \t[{}]".format(word, ', '.join(word_to_preprocessed_words[word]),
                    #                                                            ', '.join(preprocessed_words)))
                    replaced_preprocessed_words[word] = word_to_preprocessed_words[word]
                    word_to_preprocessed_words[word] = preprocessed_words

            if len(replaced_preprocessed_words) == 0:
                break

            for word, preprocessed_words in replaced_preprocessed_words.items():
                for preprocessed_word in preprocessed_words:
                    preprocessed_word_to_occurrence[preprocessed_word] -= word_to_occurrence_full[word]
                for preprocessed_word in word_to_preprocessed_words[word]:
                    if preprocessed_word not in preprocessed_word_to_occurrence:
                        preprocessed_word_to_occurrence[preprocessed_word] = word_to_occurrence_full[word]
                    else:
                        preprocessed_word_to_occurrence[preprocessed_word] += word_to_occurrence_full[word]

            self.vocab_preprocessor.register_vocabulary(
                VocabularyProxy(preprocessed_word_to_occurrence, self.vocabulary_filter))

            extra_pass_count += 1

        # Create final vocabulary by filtering stationary set of preprocessed words by occurrence criterion
        word_to_occurrence = {}
        word_to_id = {}
        new_word_id = lambda : len(word_to_id)

        word_to_occurrence['<unk>'] = 0
        word_to_id['<unk>'] = new_word_id()

        # Get special symbols from regularizer and add them to the vocabulary
        for sw in TextRegularizer.get_special_words():
            # print("Adding special symbol to vocabulary %s" % sw)
            word_to_occurrence[sw] = 0
            word_to_id[sw] = new_word_id()

        for word, occurrence in preprocessed_word_to_occurrence.items():
            if word not in word_to_id:
                if self.vocabulary_filter(word, occurrence):
                    word_to_id[word] = new_word_id()
                    word_to_occurrence[word] = preprocessed_word_to_occurrence[word]

        self._vocabulary_generated = True

        return word_to_id, word_to_occurrence


class DefaultPreprocessor:
    def __init__(self, remove_unknown_words=False):
        self.remove_unknown_words = remove_unknown_words

    def register_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary
        self.tr = TextRegularizer(vocabulary)

    def clone_shallow(self):
        preprocessor = DefaultPreprocessor(self.remove_unknown_words)
        return preprocessor

    def initial_pass_vocab(self, word):
        return self.tr.regularize_word_static(word)

    def extra_pass_vocab(self, word):
        return self.stemming_filter_preprocessing_tweet([word])

    def preprocess_tweet(self, tweet):
        token_seq = self.lexical_preprocessing_tweet(tweet)
        token_seq = self.stemming_filter_preprocessing_tweet(token_seq)
        token_seq = self.vocabulary_filtering_tweet(token_seq)
        token_seq = self.filter_unknown_words_tweet(token_seq)
        return token_seq

    ### Several preprocessing steps
    def lexical_preprocessing_tweet(self, tweet):
        """Lexical Tweet preprocessing: tokenization"""
        words = tweet.rstrip().split(' ')
        return words

    def stemming_filter_preprocessing_tweet(self, token_seq):
        """Stemming/vocabulary filtering preprocessing of tokenized tweet"""
        # Token regularization
        regularized_words = []
        for word in token_seq:
            new_word_list = self.tr.regularize_word(word)
            for new_word in new_word_list:
                regularized_words.append(new_word)

        return regularized_words

    def vocabulary_filtering_tweet(self, token_seq):
        # Replacing tokens by vocabulary terms or '<unk>' for unknown terms
        # for word in token_seq:
        #     if word not in self.vocabulary.word_to_id and len(word) > 3:
        #         print("\tUnknown: %s" % word)
        return [(word if word in self.vocabulary.word_to_id else '<unk>') for word in token_seq]

    def filter_unknown_words_tweet(self, token_seq):
        return [word for word in filter(lambda w: w != '<unk>' or not self.remove_unknown_words, token_seq)]
