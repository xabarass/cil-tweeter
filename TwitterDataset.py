import pickle
import logging
import os
import random
import numpy as np

import config

class TwitterDataSet:
    def __init__(self, should_train, positive_tweets=None,
                 negative_tweets=None,
                 test_data=None,
                 vocab_path=None,
                 test_vocab_path=None,
                 remove_unknown_words=False,
                 min_word_occ=5):

        print("Initializing...")
        # File paths
        self.pos_tw_path=positive_tweets
        self.neg_tw_path=negative_tweets
        self.vocab_path=vocab_path
        self.test_vocab_path=test_vocab_path
        self.test_data_path = test_data
        self.remove_unknown_words=remove_unknown_words
        # Train data
        self.train_tweets=[]
        self.train_sentiments=[]
        # Shuffled train data
        self.shuffled_train_tweets=[]
        self.shuffled_train_sentiments=[]
        # Test data
        self.test_tweets=[]
        # Utils data structures
        self.max_tweet_length=0
        self.word_count=1
        self.word_to_id={}
        self.min_word_occurence=min_word_occ
        # Full tweets
        self.full_tweets=[]
        self.full_train_tweets=[]
        self.full_test_tweets=[]

        self.unused_words=[]

        if self.remove_unknown_words:
            print("Removing unused words from tweets!")

        if not os.path.isfile(self.pos_tw_path):
            raise Exception("Not a valid file: %s"%self.pos_tw_path)
        if not os.path.isfile(self.neg_tw_path):
            raise Exception("Not a valid file: %s"%self.neg_tw_path)
        if not os.path.isfile(self.test_data_path):
            raise Exception("Not a valid file: %s"%self.test_data_path)
        if not os.path.isfile(self.vocab_path):
            raise Exception("Not a valid file: %s"%self.vocab_path)

        self._load_dictionary()
        self._load_data()

    def _load_dictionary(self):
        print("Loading dictionary..")
        test_vocab = {}
        with open(self.test_vocab_path,'r') as test_vocab_file:
            for line in test_vocab_file:
                tokens = line.split(' ')
                occurrence = tokens[-2]
                word = tokens[-1]
                test_vocab[word] = occurrence

        with open(self.vocab_path, 'r') as vocab:
            for line in vocab:
                tokens = line.split(' ')
                occurence = tokens[-2]
                word = tokens[-1]
                if ((int(occurence) >= self.min_word_occurence) ): # or ((int(occurrence) == self.min_word_occurence) and (word in test_vocab))) :
                    word = word.rstrip()
                    if word not in self.word_to_id:
                        self.word_to_id[word] = self.word_count
                        self.word_count= self.word_count+ 1
                else:
                    self.unused_words.append(word)
        print("Vocabulary of model has {} words".format(len(self.word_to_id)))

    # Lexical preprocessing
    def _lexical_preprocess_tweet(self,tweet):
        # instead of words = tweet.split(' ')[:]; words[-1] = words[-1].rstrip()
        words = tweet.rstrip().split(' ')
        if len(words) > self.max_tweet_length:
            self.max_tweet_length = len(words)
        return words

    # Map token sequence to id sequence
    def _map_tweet_to_id_seq(self,token_seq):
        converted_tweet = []
        for word in token_seq:
            if word in self.word_to_id:
                converted_tweet.append(self.word_to_id[word])
            elif not self.remove_unknown_words:
                converted_tweet.append(0) #Unknown word is 0

    def _lexical_preprocess_and_map_to_id_tweet(self, tweet):
        token_seq = self._lexical_preprocess_tweet(tweet)
        return self._map_tweet_to_id_seq(token_seq)

    def _add_test_tweet(self, tweet):
        self.test_tweets.append(self._lexical_preprocess_and_map_to_id_tweet(tweet))
        self.full_test_tweets.append(self._lexical_preprocess_tweet(tweet))

    def _add_train_tweet(self, tweet, sentiment):
        self.train_tweets.append(self._lexical_preprocess_and_map_to_id_tweet(tweet))
        self.train_sentiments.append(sentiment)
        self.full_train_tweets.append(self._lexical_preprocess_tweet(tweet))


    # TODO: Need to store both lexically preprocessed and not original tweets to display misclassified samples
    def _load_data(self):
        print("Loading data...")
        with open(self.pos_tw_path, 'r') as pos, open(self.neg_tw_path, 'r') as neg, open(self.test_data_path,
                                                                                                  'r') as tst:
            for line in pos:
                self._add_train_tweet(line, 1)
            for line in neg:
                self._add_train_tweet(line, 0)
            for line in tst:
                self._add_test_tweet(line)
        if config.test_run:
            self.train_tweets     = self.train_tweets[                :int(config.test_run_data_ratio*len(self.train_tweets)    )  ]
            self.train_sentiments = self.train_sentiments[            :int(config.test_run_data_ratio*len(self.train_sentiments))  ]
            self.test_tweets      = self.test_tweets[                 :int(config.test_run_data_ratio*len(self.test_tweets)     )  ]
            self.full_train_tweets      = self.full_train_tweets[     :int(config.test_run_data_ratio*len(self.full_train_tweets)) ]
            self.full_test_tweets      = self.full_test_tweets[       :int(config.test_run_data_ratio*len(self.full_test_tweets))  ]


        self.full_tweets = self.full_train_tweets + self.full_test_tweets



    def shuffle_and_split(self, split_ratio):
        print("Shuffling data...")
        nb_validation_samples = int(split_ratio * len(self.train_tweets))

        if len(self.shuffled_train_tweets) == 0 and len(self.shuffled_train_sentiments) == 0:
            combined = list(zip(self.train_tweets, self.train_sentiments))
        else:
            combined = list(zip(self.shuffled_train_tweets, self.shuffled_train_sentiments))
        random.shuffle(combined)

        self.shuffled_train_tweets[:], self.shuffled_train_sentiments[:] = zip(*combined)

        x_train = self.shuffled_train_tweets[:nb_validation_samples]
        y_train = self.shuffled_train_sentiments[:nb_validation_samples]
        x_val = self.shuffled_train_tweets[nb_validation_samples:]
        y_val = self.shuffled_train_sentiments[nb_validation_samples:]

        return (x_train, y_train), (x_val, y_val)
