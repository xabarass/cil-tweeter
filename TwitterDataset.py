import pickle
import logging
import os
import random
import numpy as np
from TextRegularizer import TextRegularizer

import config


class TwitterDataSet:
    def __init__(self,
                 positive_tweets=None,
                 negative_tweets=None,
                 test_data=None,
                 vocab_path=None,
                 test_vocab_path=None):

        print("Loading TwitterDataSet...")
        # File paths
        self.pos_tw_path=positive_tweets
        self.neg_tw_path=negative_tweets
        self.test_data_path = test_data
        self.vocab_path=vocab_path
        self.test_vocab_path=test_vocab_path
        # Train data
        self.train_sentiments=[]
        self.original_train_tweets = []
        # Test data
        self.original_test_tweets = []

        if not os.path.isfile(self.pos_tw_path):
            raise Exception("Not a valid file: %s"%self.pos_tw_path)
        if not os.path.isfile(self.neg_tw_path):
            raise Exception("Not a valid file: %s"%self.neg_tw_path)
        if not os.path.isfile(self.test_data_path):
            raise Exception("Not a valid file: %s"%self.test_data_path)
        if not os.path.isfile(self.vocab_path):
            raise Exception("Not a valid file: %s"%self.vocab_path)


        def add_test_tweet(tweet):
            self.original_test_tweets.append(tweet)

        def add_train_tweet(tweet, sentiment):
            self.train_sentiments.append(sentiment)
            self.original_train_tweets.append(tweet)

        print("Loading tweets...")
        with open(self.pos_tw_path, 'r') as pos:
            for line in pos:
                add_train_tweet(line, 1)
        with open(self.neg_tw_path, 'r') as neg:
            for line in neg:
                add_train_tweet(line, 0)
        with open(self.test_data_path,'r') as tst:
            for line in tst:
                add_test_tweet(line)

    def create_vocabulary(self, vocab_transformer):
        return Vocabulary(self.vocab_path,
                          vocab_transformer,
                          self.test_vocab_path)

    def create_preprocessed_dataset(self, vocabulary, training_validation_split_ratio):
        return PreprocessedDataset(self, vocabulary, training_validation_split_ratio)


class PreprocessedDataset:
    def __init__(self, twitter_dataset, vocabulary, training_validation_split_ratio):
        self.vocabulary = vocabulary
        self.training_validation_split_ratio = training_validation_split_ratio

        if not config.test_run:
            self.shuffled_original_train_tweets = twitter_dataset.original_train_tweets[:]
            self.shuffled_train_sentiments = twitter_dataset.train_sentiments[:]
        else:
            combined_training_dataset = list(zip(twitter_dataset.original_train_tweets, self.train_sentiments))
            random.shuffle(combined_training_dataset)
            combined_training_dataset = combined_training_dataset[:int(config.test_run_data_ratio*len(combined_training_dataset))]
            self.shuffled_original_train_tweets, self.shuffled_train_sentiments = zip(*combined_training_dataset)

        self.shuffled_train_tweets = [ vocabulary.preprocess_and_map_tweet_to_id_seq(tweet)
                                       for tweet in self.shuffled_original_train_tweets ]
        self.shuffled_preprocessed_train_tweets = [ vocabulary.map_id_seq_to_tweet(id_seq)
                                                    for id_seq in self.shuffled_train_tweets ]

        self.original_test_tweets = twitter_dataset.original_test_tweets
        self.test_tweets = [ vocabulary.preprocess_and_map_tweet_to_id_seq(tweet)
                             for tweet in self.original_test_tweets ]
        self.preprocessed_test_tweets = [ vocabulary.map_id_seq_to_tweet(id_seq)
                                          for id_seq in self.test_tweets ]

        self.all_preprocessed_tweets_randomized = self.shuffled_preprocessed_train_tweets + self.preprocessed_test_tweets

    def all_preprocessed_tweets(self):
        random.shuffle(self.all_preprocessed_tweets_randomized)
        return self.all_preprocessed_tweets_randomized

    def shuffle_and_split(self):
        """Create training data set"""
        print("Shuffling data...")

        combined_training_dataset = list(zip(self.shuffled_original_train_tweets, self.shuffled_train_tweets, self.shuffled_train_sentiments))
        random.shuffle(combined_training_dataset)
        self.shuffled_original_train_tweets, self.shuffled_train_tweets, self.shuffled_train_sentiments = zip(*combined_training_dataset)

        # TODO: Remove implicit config.validation_split_ratio dependency
        nb_validation_samples = int(self.training_validation_split_ratio * len(self.shuffled_train_tweets))

        x_train = self.shuffled_train_tweets[:nb_validation_samples]
        x_orig_train = self.shuffled_original_train_tweets[:nb_validation_samples]
        y_train = self.shuffled_train_sentiments[:nb_validation_samples]
        x_val   = self.shuffled_train_tweets[nb_validation_samples:]
        x_orig_val   = self.shuffled_original_train_tweets[nb_validation_samples:]
        y_val = self.shuffled_train_sentiments[nb_validation_samples:]

        return (x_train, y_train, x_orig_train), (x_val, y_val, x_orig_val)


class Vocabulary:
    def __init__(self, vocab_path, vocab_transformer, test_vocab_path=None):
        """Load and compute vocabulary from output of twitter-datasets/build_vocab.sh"""
        if vocab_transformer is None:
             raise Exception("Not a valid vocabulary transformer supplied: %s" % repr(vocab_transformer))

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
        for word in self.word_to_id:
            self.id_to_word[self.word_to_id[word]] = word

        print("Vocabulary of model has {} words".format(len(self.word_to_id)))

        self.max_tweet_length = 0

    def preprocess_and_map_tweet_to_id_seq(self, tweet):
        """Preprocess tweet with lexical/stemming/filtering phase and replace every token by vocabulary integer id"""
        assert isinstance(tweet,str)
        token_seq = self.preprocessor.preprocess_tweet(tweet)
        return self.map_to_id_seq_tweet(token_seq)

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
        self.min_word_occurence = min_word_occurence
        self.remove_unknown_words =remove_unknown_words

    def register_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary
        self.tr = TextRegularizer(vocabulary)

    # TODO: Filter some of the very short and relatively rare words here <5-10 occurrences for length 3, <15-30 for length 2
    def first_pass_vocab(self, word,occurrence):
        return (occurrence >= self.min_word_occurence), self.tr.regularize_word_vocab(word) # Note we should actually accumulate all of the regularized words

    # # FIXME: Do we actually need a second pass? The hashtags normally do not add any further words...
    # def second_pass_vocab(self, word, occurrence, word_to_occurrence):
    #     return (occurrence >= self.min_word_occurence), [new_word for new_word in filter(lambda w: w not in self.vocabulary.word_to_id,
    #                                                   self.tr.regularize_word_vocab(word, word_to_occurrence))]

    # TODO: remove max_tweet_length modificaiton
    def preprocess_tweet(self, tweet):
        token_seq = self.lexical_preprocessing_tweet(tweet)
        token_seq = self.stemming_filter_preprocessing_tweet(token_seq)
        token_seq = self.filter_unknown_words(token_seq)
        if len(token_seq) > self.vocabulary.max_tweet_length:
            self.vocabulary.max_tweet_length = len(token_seq)
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
            new_word_list = tr.regularize_word(word)
            for new_word in new_word_list:
                regularized_words.append(new_word)

        # Replacing tokens by vocabulary terms or '<unk>' for unknown terms
        return [ (word if word in self.word_to_id else '<unk>')  for word in regularized_words ]

    def filter_unknown_words(self, token_seq):
        # TODO: remove config dependency
        return [word for word in filter(lambda w: w != '<unk>' or not self.remove_unknown_words, token_seq)]
