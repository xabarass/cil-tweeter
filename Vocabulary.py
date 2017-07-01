import pickle
import logging
import os
import numpy as np
from tqdm import tqdm
from TextRegularizer import TextRegularizer


def read_vocabulary_from_file(vocab_path, test_vocab_path=None):
    """Load vocabulary from output file of twitter-datasets/build_vocab.sh"""
    if not os.path.isfile(vocab_path):
        raise Exception("Not a valid file: %s" % vocab_path)

    print("Loading vocabulary from file %s.." % vocab_path)

    # # Load test vocabulary
    # if test_vocab_path is not None:
    #     test_vocab = {}
    #     with open(test_vocab_path, 'r') as test_vocab_file:
    #         for line in test_vocab_file:
    #             tokens = line.split(' ')
    #             occurrence = int(tokens[-2])
    #             word = tokens[-1]
    #             assert word not in test_vocab
    #             test_vocab[word] = occurrence

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

    return word_to_occurrence


class Vocabulary:
    """Creates an internal preprocessor vocabulary from unfiltered word-frequency dict"""
    def __init__(self, preprocessed_word_to_occurrence, preprocessor):
        self.preprocessor = preprocessor

        if hasattr(preprocessor.final_vocabulary_filter, "min_word_occurrence"):
            self.min_word_occurrence = preprocessor.final_vocabulary_filter.min_word_occurrence
        else:
            raise

        self.word_to_id, self.word_to_occurrence = create_filtered_vocabulary(preprocessed_word_to_occurrence,
                                                                              preprocessor.get_special_symbols(),
                                                                              preprocessor.final_vocabulary_filter)

        self.filter = preprocessor.final_vocabulary_filter

        # build reverse lookup dictionary
        self.id_to_word = {}
        for word in self.word_to_id:
            self.id_to_word[self.word_to_id[word]] = word

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

    def map_tweet_to_id_seq(self, tweet_token_seq):
        """Replace every token by vocabulary integer id"""
        assert isinstance(tweet_token_seq, list)
        return [self.word_to_id[word] for word in tweet_token_seq]

    def map_id_seq_to_tweet(self, tweet_id_seq):
        """Replace every vocabulary integer id by token"""
        assert isinstance(tweet_id_seq, list)
        return [self.id_to_word[id] for id in tweet_id_seq]

class InternalVocabulary:
    """Creates an stripped down vocabulary for internal use by the preprocessor based on unfiltered word-frequency dict"""

    def __init__(self, unfiltered_word_to_occurrence, preprocessor):
        _, self.word_to_occurrence = create_filtered_vocabulary(unfiltered_word_to_occurrence,
                                                                preprocessor.get_special_symbols(),
                                                                preprocessor.internal_vocabulary_filter)

        self.filter = preprocessor.internal_vocabulary_filter



def create_filtered_vocabulary(preprocessed_word_to_occurrence,
                               special_symbols,
                               vocabulary_filter):
    """Creates a vocabulary (word enumeration and occurrence dicts) from candidate set filtered with a vocabulary_filter
       predicate and additional special symbols"""
    word_to_occurrence = {}
    word_to_id = {}
    new_word_id = lambda : len(word_to_id)

    # Add special symbols to the vocabulary first
    for sw in special_symbols:
        # print("Adding special symbol to vocabulary %s" % sw)
        word_to_occurrence[sw] = 0
        word_to_id[sw] = new_word_id()

    for word, occurrence in preprocessed_word_to_occurrence.items():
        if word not in word_to_id:
            if vocabulary_filter(word, occurrence):
                word_to_id[word] = new_word_id()
                word_to_occurrence[word] = preprocessed_word_to_occurrence[word]

    return word_to_id, word_to_occurrence


def iterative_vocabulary_generator(word_to_occurrence_full, preprocessor):
    """Function object that creates word_to_id vocabulary dictionary from word_to_occurrence statistics of corpus"""

    # Two passes of vocabulary creation
    # First collect words by frequency, in a second phase create additional words
    preprocessor._update_vocabulary( unfiltered_word_to_occurrence={} )

    word_to_preprocessed_words, preprocessed_word_to_occurrence = \
        initial_pass_vocabulary(word_to_occurrence_full=word_to_occurrence_full,
                                tokenizer=preprocessor.initial_pass_vocab)

    preprocessor._update_vocabulary( preprocessed_word_to_occurrence )

    extra_pass_count = 0
    while True:
        replaced_preprocessed_words = extra_pass_vocabulary(
                              word_to_occurrence_full=word_to_occurrence_full,
                              tokenizer=preprocessor.extra_pass_vocab,
                              word_to_preprocessed_words=word_to_preprocessed_words,
                              preprocessed_word_to_occurrence=preprocessed_word_to_occurrence,
                              extra_pass_count=extra_pass_count+1)

        preprocessor._update_vocabulary( preprocessed_word_to_occurrence )

        print("[VocabularyGen] - extra_pass %d: Updated %d tokenizations" % (extra_pass_count+1, len(replaced_preprocessed_words)))
        if len(replaced_preprocessed_words) == 0:
            break

        extra_pass_count += 1

    # Create final vocabulary by filtering stationary set of preprocessed words by occurrence criterion
    return preprocessed_word_to_occurrence


def single_pass_vocabulary_generator(word_to_occurrence_full, preprocessor):
    """Function object that creates word_to_id vocabulary dictionary from word_to_occurrence statistics of corpus"""

    # Single pass of vocabulary creation
    word_to_preprocessed_words, preprocessed_word_to_occurrence = \
        initial_pass_vocabulary(word_to_occurrence_full=word_to_occurrence_full,
                                tokenizer=preprocessor.initial_pass_vocab)

    # Create final vocabulary by filtering set of preprocessed words by occurrence criterion
    return preprocessed_word_to_occurrence


def initial_pass_vocabulary(word_to_occurrence_full, tokenizer):
    word_to_preprocessed_words = {}
    preprocessed_word_to_occurrence = {}

    for word in tqdm(word_to_occurrence_full, desc="[VocabularyGen] - initial pass (token generation)"):
        preprocessed_words = tokenizer(word)  # preprocessor returns a list of words that word parameter gets preprocessed into
        assert isinstance(preprocessed_words, list)
        word_to_preprocessed_words[word] = preprocessed_words
        # Compute preprocessed token -> frequency dict
        for preprocessed_word in preprocessed_words:
            if preprocessed_word not in preprocessed_word_to_occurrence:
                preprocessed_word_to_occurrence[preprocessed_word] = word_to_occurrence_full[word]
            else:
                preprocessed_word_to_occurrence[preprocessed_word] += word_to_occurrence_full[word]

    return word_to_preprocessed_words, preprocessed_word_to_occurrence


def extra_pass_vocabulary(word_to_occurrence_full,
                      tokenizer,
                      word_to_preprocessed_words,
                      preprocessed_word_to_occurrence,
                      extra_pass_count):

    replaced_preprocessed_words = {}
    # print("\t[DefaultVocabularyTransformer]\t - %d-th extra pass" % (extra_pass_count+1))
    for word in tqdm(word_to_occurrence_full,desc="[VocabularyGen] - extra pass %d: Updating tokenizations" % (extra_pass_count)):
        preprocessed_words = tokenizer(word)  # preprocessor returns a list of words that word parameter gets preprocessed into
        assert isinstance(preprocessed_words, list)
        if preprocessed_words != word_to_preprocessed_words[word]:
            # print("\t '{}':\n\t\t[{}]\t --> \t[{}]".format(word, ', '.join(word_to_preprocessed_words[word]),
            #                                                            ', '.join(preprocessed_words)))
            replaced_preprocessed_words[word] = word_to_preprocessed_words[word]
            word_to_preprocessed_words[word] = preprocessed_words

            # correct tokenization frequencies
            for preprocessed_word in replaced_preprocessed_words[word]:
                preprocessed_word_to_occurrence[preprocessed_word] -= word_to_occurrence_full[word]

            for preprocessed_word in preprocessed_words:
                if preprocessed_word not in preprocessed_word_to_occurrence:
                    preprocessed_word_to_occurrence[preprocessed_word] = word_to_occurrence_full[word]
                else:
                    preprocessed_word_to_occurrence[preprocessed_word] += word_to_occurrence_full[word]

    return replaced_preprocessed_words



class BasePreprocessor:
    def __init__(self, final_vocabulary_filter,  remove_unknown_words):
        self.remove_unknown_words = remove_unknown_words
        self.final_vocabulary_filter = final_vocabulary_filter

    @property
    def vocabulary(self):
        return self.final_vocabulary


    def get_special_symbols(self):
        return (['<pad>'] if self.remove_unknown_words else ['<pad>','<unk>']) + self._get_special_symbols()

    def preprocess_tweet(self, tweet):
        token_seq = self.lexical_preprocessing_tweet(tweet)
        token_seq = self.secondary_preprocessing_tweet(token_seq)
        token_seq = self.vocabulary_filtering_tweet(token_seq)
        return token_seq

    def lexical_preprocessing_tweet(self, tweet):
        """Lexical Tweet preprocessing: tokenization"""
        raise Exception("To be implemented by derived class")

    def secondary_preprocessing_tweet(self, token_seq):
        raise Exception("To be implemented by derived class")

    def vocabulary_filtering_tweet(self, token_seq):
        # Replacing tokens by vocabulary terms or '<unk>' for unknown terms if self.remove_unknown_words != True
        token_seq = [(word if word in self.final_vocabulary.word_to_id else '<unk>') for word in token_seq]
        return [word for word in filter(lambda w: w != '<unk>' or not self.remove_unknown_words, token_seq)]


    def preprocess_and_map_tweet_to_id_seq(self, tweet):
        """Preprocess tweet with lexical/stemming/filtering phase and replace every token by vocabulary integer id"""
        assert isinstance(tweet, str)
        token_seq = self.preprocess_tweet(tweet)
        return self.final_vocabulary.map_tweet_to_id_seq(token_seq)

    def map_tweet_to_id_seq(self, tweet_token_seq):
        """Replace every token by vocabulary integer id"""
        return self.final_vocabulary.map_tweet_to_id_seq(tweet_token_seq=tweet_token_seq)

    def map_id_seq_to_tweet(self, tweet_id_seq):
        """Replace every vocabulary integer id by token"""
        return self.final_vocabulary.map_id_seq_to_tweet(tweet_id_seq=tweet_id_seq)


class LexicalPreprocessor(BasePreprocessor):
    def __init__(self, word_to_occurrence_full, final_vocabulary_filter, remove_unknown_words=False):
        super(LexicalPreprocessor,self).__init__(final_vocabulary_filter=final_vocabulary_filter, remove_unknown_words=remove_unknown_words)

        if not isinstance(word_to_occurrence_full, dict):
            raise Exception("Must deal separately with vocabulary provided in a list")
        preprocessed_word_to_occurrence = \
            single_pass_vocabulary_generator(word_to_occurrence_full=word_to_occurrence_full,
                                             preprocessor=self)
        self.final_vocabulary = Vocabulary(preprocessed_word_to_occurrence=preprocessed_word_to_occurrence,
                                           preprocessor=self)
        print("Vocabulary of model has {} words".format(self.final_vocabulary.word_count))

    def _get_special_symbols(self):
        return []

    def initial_pass_vocab(self, word):
        return self.secondary_preprocessing_tweet(word)

    @classmethod
    def lexical_preprocessing_tweet(cls, tweet):
        """Lexical tweet preprocessing - tokenization"""
        words = tweet.rstrip().split(' ')
        return words

    def secondary_preprocessing_tweet(self, token_seq):
        """Stemming preprocessing of tokenized tweet"""
        # Token regularization
        return token_seq


class RegularizingPreprocessor(BasePreprocessor):
    def __init__(self, word_to_occurrence_full, final_vocabulary_filter, preprocessor_vocabulary_filter, remove_unknown_words=False):
        super(RegularizingPreprocessor,self).__init__(final_vocabulary_filter=final_vocabulary_filter,remove_unknown_words=remove_unknown_words)
        self.internal_vocabulary_filter=preprocessor_vocabulary_filter

        if not isinstance(word_to_occurrence_full, dict):
            raise Exception("Must deal separately with vocabulary provided in a list")
        self._update_vocabulary({})
        preprocessed_word_to_occurrence = iterative_vocabulary_generator(word_to_occurrence_full=word_to_occurrence_full,
                                                                         preprocessor=self)
        self._update_vocabulary(preprocessed_word_to_occurrence)
        print("Vocabulary of model has {} words".format(self.final_vocabulary.word_count))

    def _update_vocabulary(self, unfiltered_word_to_occurrence):
        self.internal_vocabulary = InternalVocabulary(unfiltered_word_to_occurrence,self)
        self.final_vocabulary    = Vocabulary(unfiltered_word_to_occurrence, self)
        self.tr = TextRegularizer(final_vocabulary=self.final_vocabulary,
                                  internal_vocabulary=self.internal_vocabulary)

    def _get_special_symbols(self):
        return TextRegularizer.get_special_words()

    def initial_pass_vocab(self, word):
        return self.tr.regularize_word_static(word)

    def extra_pass_vocab(self, word):
        return self.secondary_preprocessing_tweet([word])

    @classmethod
    def lexical_preprocessing_tweet(cls, tweet):
        """Lexical tweet preprocessing - tokenization"""
        words = tweet.rstrip().split(' ')
        return words

    def secondary_preprocessing_tweet(self, token_seq):
        """Stemming filtering preprocessing of tokenized tweet"""
        # Token regularization
        regularized_words = []
        for word in token_seq:
            new_word_list = self.tr.regularize_word(word)
            for new_word in new_word_list:
                regularized_words.append(new_word)

        return regularized_words
