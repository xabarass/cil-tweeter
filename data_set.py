import pickle
import logging
import os
import random
import numpy as np

class TwitterDataSet:
    def __init__(self, should_train, positive_tweets=None,
                 negative_tweets=None,
                 test_data=None,
                 vocab_path=None,
                 remove_unknown_words=False):

        print("Initializing...")
        # File paths
        self.pos_tw_path=positive_tweets
        self.neg_tw_path=negative_tweets
        self.vocab_path=vocab_path
        self.test_data_path = test_data
        self.remove_unknown_words=remove_unknown_words
        # Train data
        self.train_tweets=[]
        self.train_sentiments=[]
        # Test data
        self.test_tweets=[]
        # Utils data structures
        self.max_tweet_length=0
        self.word_count=1
        self.word_to_id={}
        self.min_word_occurence=5
        # Full tweets
        self.full_tweets=[]

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
        with open(self.vocab_path, 'r') as vocab:
            for line in vocab:
                tokens = line.split(' ')
                occurence = tokens[-2]
                word = tokens[-1]
                if int(occurence) >= self.min_word_occurence:
                    word = word.rstrip()
                    if word not in self.word_to_id:
                        self.word_to_id[word] = self.word_count
                        self.word_count= self.word_count+ 1
                else:
                    self.unused_words.append(word)

    def _convert_tweet(self, tweet):
        words = tweet.split(' ')[:]
        if len(words) > self.max_tweet_length:
            self.max_tweet_length = len(words)

        words[-1] = words[-1].rstrip()

        converted_tweet = []
        for word in words:
            if word in self.word_to_id:
                converted_tweet.append(self.word_to_id[word])
            elif not self.remove_unknown_words:
                converted_tweet.append(0) #Unknown word is 0

        return converted_tweet

    def _add_full_tweet(self, tweet):
        self.full_tweets.append(tweet.rstrip().split(' '))

    def _add_test_data(self, tweet):
        self.test_tweets.append(self._convert_tweet(tweet))

    def _add_tweet(self, tweet, sentiment):
        self.train_tweets.append(self._convert_tweet(tweet))
        self.train_sentiments.append(sentiment)

    def _load_data(self):
        print("Loading data...")
        with open(self.pos_tw_path, 'r') as pos, open(self.neg_tw_path, 'r') as neg, open(self.test_data_path,
                                                                                                  'r') as tst:
            for line in pos:
                self._add_tweet(line, 1)
                self._add_full_tweet(line)
            for line in neg:
                self._add_tweet(line, 0)
                self._add_full_tweet(line)
            for line in tst:
                self._add_test_data(line)
                self._add_full_tweet(line)

    def shuffle_and_split(self, split_ratio):
        print("Shuffling data...")
        nb_validation_samples = int(split_ratio * len(self.train_tweets))

        combined = list(zip(self.train_tweets, self.train_sentiments))
        random.shuffle(combined)

        self.train_tweets[:], self.train_sentiments[:] = zip(*combined)

        x_train = self.train_tweets[:nb_validation_samples]
        y_train = self.train_sentiments[:nb_validation_samples]
        x_val = self.train_tweets[nb_validation_samples:]
        y_val = self.train_sentiments[nb_validation_samples:]

        return (x_train, y_train), (x_val, y_val)
