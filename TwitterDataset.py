import pickle
import logging
import os
import random
import numpy as np
from tqdm import tqdm
from keras.preprocessing import sequence

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


# TODO: For character embeddings pass in a character vocabulary (mapping tweets to id sequences)
class PreprocessedDataset:
    def __init__(self, twitter_dataset, vocabulary, training_validation_split_ratio):
        assert vocabulary is not None
        self.vocabulary = vocabulary

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

        print("[TrainingDataset] Preprocessing training and test tweets...")
        if self.vocabulary is not None:
            self.shuffled_train_tweets = []
            for tweet in tqdm(self.shuffled_original_train_tweets, desc="Training tweets"):
                self.shuffled_train_tweets.append(vocabulary.preprocess_and_map_tweet_to_id_seq(tweet))
        else:
            self.shuffled_train_tweets = self.shuffled_original_train_tweets[:]

        self.original_test_tweets = twitter_dataset.original_test_tweets

        if self.vocabulary is not None:
            self.test_tweets = []
            for tweet in tqdm(self.original_test_tweets, desc="Test tweets"):
                self.test_tweets.append(vocabulary.preprocess_and_map_tweet_to_id_seq(tweet))
        else:
            self.test_tweets = self.original_test_tweets[:]

        # Store longest tweet length (as id sequence)
        self.max_tweet_length = 0
        for id_seq in self.shuffled_train_tweets + self.test_tweets:
            if len(id_seq) > self.max_tweet_length:
                self.max_tweet_length = len(id_seq)

        print("[TrainingDataset] Max tweet length: %d" % self.max_tweet_length)


        self.shuffled_preprocessed_train_tweets = []
        for id_seq in tqdm(self.shuffled_train_tweets,desc="Training tweets: generate tokens from id seq"):
            self.shuffled_preprocessed_train_tweets.append( vocabulary.map_id_seq_to_tweet(id_seq) )

        self.preprocessed_test_tweets = []
        for id_seq in tqdm(self.test_tweets, desc="Test tweets: generate tokens from id seq"):
            self.preprocessed_test_tweets.append( vocabulary.map_id_seq_to_tweet(id_seq) )

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

    def shuffle_and_split(self):
        """Create training data set"""
        print("[TrainingDataset] Shuffling data...")

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


    def shuffle_and_split_padded(self):
        (x_train, y_train, x_orig_train), (x_val, y_val, x_orig_val) = self.shuffle_and_split()
        x_train = sequence.pad_sequences(x_train, maxlen=self.max_tweet_length, value=self.vocabulary.word_to_id['<pad>'] )
        x_val   = sequence.pad_sequences(x_val,   maxlen=self.max_tweet_length, value=self.vocabulary.word_to_id['<pad>'] )
        return (x_train, y_train, x_orig_train), (x_val, y_val, x_orig_val)


    def test_tweets_padded(self):
        return sequence.pad_sequences(self.test_tweets, maxlen=self.max_tweet_length, value=self.vocabulary.word_to_id['<pad>'])
