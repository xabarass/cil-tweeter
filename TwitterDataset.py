import pickle
import logging
import os
import random
import numpy as np

import config


class TwitterDataSet:
    def __init__(self,
                 positive_tweets=None,
                 negative_tweets=None,
                 test_data=None):

        print("Loading TwitterDataSet...")
        # File paths
        self.pos_tw_path=positive_tweets
        self.neg_tw_path=negative_tweets
        self.test_data_path = test_data
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

    # TODO: possibly move this elsewhere
    def create_preprocessed_dataset(self, vocabulary, training_validation_split_ratio):
        return PreprocessedDataset(self, vocabulary, training_validation_split_ratio)

    def create_unpreprocessed_dataset(self, training_validation_split_ratio):
        return UnpreprocessedDataset(self, training_validation_split_ratio)


class TrainingDataset:
    def __init__(self, twitter_dataset, training_validation_split_ratio, preprocess_tweet=None):
        self.training_validation_split_ratio = training_validation_split_ratio

        self.shuffled_original_train_tweets = []
        self.shuffled_train_sentiments = []
        if not config.test_run:
            self.shuffled_original_train_tweets[:] = twitter_dataset.original_train_tweets[:]
            self.shuffled_train_sentiments[:] = twitter_dataset.train_sentiments[:]
        else:
            combined_training_dataset = list(
                zip(twitter_dataset.original_train_tweets[:], twitter_dataset.train_sentiments[:]))
            random.shuffle(combined_training_dataset)
            combined_training_dataset = combined_training_dataset[
                                        :int(config.test_run_data_ratio * len(combined_training_dataset))]
            self.shuffled_original_train_tweets[:], self.shuffled_train_sentiments[:] = zip(*combined_training_dataset)

        if preprocess_tweet is not None:
            self.shuffled_train_tweets = [preprocess_tweet(tweet)
                                          for tweet in self.shuffled_original_train_tweets]
        else:
            self.shuffled_train_tweets = self.shuffled_original_train_tweets[:]

        self.original_test_tweets = twitter_dataset.original_test_tweets

        if preprocess_tweet is not None:
            self.test_tweets = [preprocess_tweet(tweet)
                                for tweet in self.original_test_tweets]
        else:
            self.test_tweets = self.original_test_tweets[:]

        # Store longest tweet length (as id sequence)
        self.max_tweet_length = 0
        for id_seq in self.shuffled_train_tweets + self.test_tweets:
            if len(id_seq) > self.max_tweet_length:
                self.max_tweet_length = len(id_seq)

        print("Max tweet length: %d" % self.max_tweet_length)

    def shuffle_and_split(self):
        """Create training data set"""
        print("Shuffling data...")

        combined_training_dataset = list(
            zip(self.shuffled_original_train_tweets, self.shuffled_train_tweets, self.shuffled_train_sentiments))
        random.shuffle(combined_training_dataset)
        self.shuffled_original_train_tweets[:], self.shuffled_train_tweets[:], self.shuffled_train_sentiments[:] = zip(
            *combined_training_dataset)

        nb_validation_samples = int(self.training_validation_split_ratio * len(self.shuffled_train_tweets))

        x_train = self.shuffled_train_tweets[:nb_validation_samples]
        x_orig_train = self.shuffled_original_train_tweets[:nb_validation_samples]
        y_train = self.shuffled_train_sentiments[:nb_validation_samples]
        x_val = self.shuffled_train_tweets[nb_validation_samples:]
        x_orig_val = self.shuffled_original_train_tweets[nb_validation_samples:]
        y_val = self.shuffled_train_sentiments[nb_validation_samples:]

        return (x_train, y_train, x_orig_train), (x_val, y_val, x_orig_val)


class PreprocessedDataset(TrainingDataset):
    def __init__(self, twitter_dataset, vocabulary, training_validation_split_ratio):
        self.vocabulary = vocabulary

        super(PreprocessedDataset,self).__init__(twitter_dataset,training_validation_split_ratio,vocabulary.preprocess_and_map_tweet_to_id_seq)

        self.shuffled_preprocessed_train_tweets = [ vocabulary.map_id_seq_to_tweet(id_seq)
                                                    for id_seq in self.shuffled_train_tweets ]

        self.preprocessed_test_tweets = [ vocabulary.map_id_seq_to_tweet(id_seq)
                                          for id_seq in self.test_tweets ]

        self.all_preprocessed_tweets_randomized = self.shuffled_preprocessed_train_tweets + self.preprocessed_test_tweets

    def all_preprocessed_tweets(self):
        random.shuffle(self.all_preprocessed_tweets_randomized)
        return self.all_preprocessed_tweets_randomized

    def all_tokenized_tweets(self):
        tokenized_tweets_randomized  = [self.vocabulary.preprocessor.lexical_preprocessing_tweet(tweet)
                                        for tweet in self.shuffled_original_train_tweets]
        tokenized_tweets_randomized += [self.vocabulary.preprocessor.lexical_preprocessing_tweet(tweet)
                                        for tweet in self.original_test_tweets]
        random.shuffle(tokenized_tweets_randomized)
        return tokenized_tweets_randomized

class UnpreprocessedDataset(TrainingDataset):
    def __init__(self, twitter_dataset, training_validation_split_ratio):
        super(UnpreprocessedDataset,self).__init__(twitter_dataset,training_validation_split_ratio)
