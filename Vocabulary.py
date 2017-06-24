import pickle
import logging
import os
import numpy as np
from TextRegularizer import TextRegularizer


class Vocabulary:
    def __init__(self, vocab_transformer, vocab_path, test_vocab_path=None):
        """Load and compute vocabulary from output of twitter-datasets/build_vocab.sh"""
        if vocab_transformer is None:
             raise Exception("Not a valid vocabulary transformer supplied: %s" % repr(vocab_transformer))
        if not os.path.isfile(vocab_path):
            raise Exception("Not a valid file: %s" % vocab_path)

        # Refer to vocab to enable online tweet preprocessing from within the preprocessor
        self.preprocessor = vocab_transformer.preprocessor
        vocab_transformer.preprocessor.register_vocabulary(self)

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
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self._vocabulary_generated = False

    def __call__(self, word_to_occurrence_full):
        assert not self._vocabulary_generated

        word_to_id = {}
        new_word_id = lambda : len(word_to_id)
        word_to_occurrence = {}
        word_to_id['<unk>'] = new_word_id()

        # Get special symbols from regularizer and add them to the vocabulary
        for sw in TextRegularizer.get_special_words():
            # print("Adding special symbol to vocabulary %s" % sw)
            word_to_id[sw] = new_word_id()
            word_to_occurrence[sw] = 0

        # Two passes of vocabulary creation
        # First create all standard words, secondly add non-standard specially treated words
        #word_processed = set()
        for word in word_to_occurrence_full:
            word_in_vocab, preprocessed_words = \
                self.preprocessor.first_pass_vocab(word, word_to_occurrence_full[word])
            assert isinstance(word_in_vocab,bool)
            assert isinstance(preprocessed_words,list)
            if word_in_vocab:
                #word_processed.add(word)
                for preprocessed_word in preprocessed_words:
                    assert isinstance(preprocessed_word, str)
                    if preprocessed_word not in word_to_id:
                        word_to_id[preprocessed_word] = new_word_id()
                        word_to_occurrence[preprocessed_word] = word_to_occurrence_full[word]
                    else:
                        word_to_occurrence[preprocessed_word] += word_to_occurrence_full[word]
            #else:
            #    unused_words_first_pass.append(word)

        # for word in word_to_occurrence_full:
        #     if word not in word_processed:
        #         if word not in word_to_id:
        #             word_in_vocab, preprocessed_words = \
        #                 self.preprocessor.second_pass_vocab(word, word_to_occurrence_full[word], word_to_occurrence)
        #             assert isinstance(word_in_vocab,bool)
        #             assert isinstance(preprocessed_words,list)
        #             if word_in_vocab:
        #                 for preprocessed_word in preprocessed_words:
        #                     assert isinstance(preprocessed_word,str)
        #                     if preprocessed_word not in word_to_id:
        #                         word_to_id[preprocessed_word] = new_word_id()
        #                         word_to_occurrence[preprocessed_word] = word_to_occurrence_full[word]
        #                     else:
        #                         word_to_occurrence[preprocessed_word] += word_to_occurrence_full[word]
        #             #else:
        #             #    unused_words_second_pass.append(word)

        self._vocabulary_generated = True

        return word_to_id, word_to_occurrence


class DefaultPreprocessor:
    def __init__(self, min_word_occurrence=5, remove_unknown_words=False):
        self.min_word_occurrence = min_word_occurrence
        self.remove_unknown_words =remove_unknown_words

    def register_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary
        self.tr = TextRegularizer(vocabulary)

    # TODO: Filter some of the very short and relatively rare words here <5-10 occurrences for length 3, <15-30 for length 2
    def first_pass_vocab(self, word,occurrence):
        return (occurrence >= self.min_word_occurrence), self.tr.regularize_word_vocab(word) # Note we should actually accumulate all of the regularized words

    # # FIXME: Do we actually need a second pass? The hashtags normally do not add any further words...
    # def second_pass_vocab(self, word, occurrence, word_to_occurrence):
    #     return (occurrence >= self.min_word_occurence), [new_word for new_word in filter(lambda w: w not in self.vocabulary.word_to_id,
    #                                                   self.tr.regularize_word_vocab(word, word_to_occurrence))]

    # TODO: remove max_tweet_length modificaiton
    def preprocess_tweet(self, tweet):
        token_seq = self.lexical_preprocessing_tweet(tweet)
        token_seq = self.stemming_filter_preprocessing_tweet(token_seq)
        token_seq = self.filter_unknown_words(token_seq)
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

        # Replacing tokens by vocabulary terms or '<unk>' for unknown terms
        return [ (word if word in self.vocabulary.word_to_id else '<unk>')  for word in regularized_words ]

    def filter_unknown_words(self, token_seq):
        # TODO: remove config dependency
        return [word for word in filter(lambda w: w != '<unk>' or not self.remove_unknown_words, token_seq)]
