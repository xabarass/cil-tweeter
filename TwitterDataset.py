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
                 test_data=None,
                 deduplicate_train_tweets=False):

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
            pos_tweet_set = set()
            for i, line in enumerate(pos): # Note: i could be used to store subsequent permutations
                if deduplicate_train_tweets and (line in pos_tweet_set):
                    continue
                add_train_tweet(line, 1)
                pos_tweet_set.add(line)
        with open(self.neg_tw_path, 'r') as neg:
            neg_tweet_set = set()
            for i, line in enumerate(neg): # Note: i could be used to store subsequent permutations
                if deduplicate_train_tweets and (line in neg_tweet_set):
                    continue
                add_train_tweet(line, 0)
                neg_tweet_set.add(line)
        with open(self.test_data_path,'r') as tst:
            for line in tst:
                add_test_tweet(line)

        if deduplicate_train_tweets:
            print("[TwitterDataset] Found %d positive and %d negative tweets" % (len(pos_tweet_set),len(neg_tweet_set)))

    # TODO: possibly move this elsewhere
    def create_preprocessed_dataset(self, preprocessor, training_validation_split_ratio):
        return PreprocessedDataset(self, preprocessor, training_validation_split_ratio)


def numpy_random_shuffle(training_list):
    return [training_list[id] for id in
            np.random.permutation(np.arange(len(training_list)))]


# TODO: For character embeddings pass in a character vocabulary (mapping tweets to id sequences)
class PreprocessedDataset:
    def __init__(self, twitter_dataset, preprocessor, training_validation_split_ratio):
        assert preprocessor is not None
        self.preprocessor = preprocessor

        self.training_validation_split_ratio = training_validation_split_ratio

        self.shuffled_original_train_tweets = []
        self.shuffled_train_sentiments = []
        if not config.test_run:
            self.shuffled_original_train_tweets[:] = twitter_dataset.original_train_tweets[:]
            self.shuffled_train_sentiments[:] = twitter_dataset.train_sentiments[:]
            self.original_test_tweets = twitter_dataset.original_test_tweets
        else:
            combined_training_dataset = list(
                zip(twitter_dataset.original_train_tweets[:], twitter_dataset.train_sentiments[:]))
            #random.shuffle(combined_training_dataset)
            combined_training_dataset = numpy_random_shuffle(combined_training_dataset)
            combined_training_dataset = combined_training_dataset[
                                        :int(config.test_run_data_ratio * len(combined_training_dataset))]
            self.shuffled_original_train_tweets[:], self.shuffled_train_sentiments[:] = zip(*combined_training_dataset)
            self.original_test_tweets = twitter_dataset.original_test_tweets[:int(config.test_run_test_data_ratio *
                                                                                  len(twitter_dataset.original_test_tweets))]

        print("[TrainingDataset] Preprocessing training and test tweets...")
        self.shuffled_train_tweets = []
        for tweet in tqdm(self.shuffled_original_train_tweets, desc="Training tweets"):
            self.shuffled_train_tweets.append(preprocessor.preprocess_and_map_tweet_to_id_seq(tweet))

        self._test_tweets = []
        for tweet in tqdm(self.original_test_tweets, desc="Test tweets"):
            self._test_tweets.append(preprocessor.preprocess_and_map_tweet_to_id_seq(tweet))

        # Store longest tweet length (as id sequence)
        self.max_tweet_length = 0
        for id_seq in self.shuffled_train_tweets + self._test_tweets:
            if len(id_seq) > self.max_tweet_length:
                self.max_tweet_length = len(id_seq)

        print("[TrainingDataset] Max tweet length: %d" % self.max_tweet_length)

        self.shuffled_preprocessed_train_tweets = []
        for id_seq in tqdm(self.shuffled_train_tweets, desc="Training tweets: generate tokens from id seq"):
            self.shuffled_preprocessed_train_tweets.append(self.preprocessor.map_id_seq_to_tweet(id_seq))

        self.preprocessed_test_tweets = []
        for id_seq in tqdm(self._test_tweets, desc="Test tweets: generate tokens from id seq"):
            self.preprocessed_test_tweets.append(self.preprocessor.map_id_seq_to_tweet(id_seq))

    def all_preprocessed_tweets_weighted(self, test_weight=None):
        #random.shuffle(self.all_preprocessed_tweets_randomized)
        return numpy_random_shuffle(self.shuffled_preprocessed_train_tweets + (self.preprocessed_test_tweets
                                                                               if (test_weight is None) else
                                                                               self.preprocessed_test_tweets * test_weight) )

    def weighted_preprocessed_tweets(self, preprocessed_train_tweets, sample_weight, test_weight=None):
        print("[PreprocessedDataset] Getting weighted preprocessed tweets using supplied train tweets...")

        shuffled_preprocessed_train_tweets = []
        for tweet, weight in zip(preprocessed_train_tweets, sample_weight):
            shuffled_preprocessed_train_tweets += tweet * int(np.ceil(weight))

        #random.shuffle(self.all_preprocessed_tweets_randomized)
        return numpy_random_shuffle(shuffled_preprocessed_train_tweets +  (self.preprocessed_test_tweets
                                                                           if (test_weight is None) else
                                                                           self.preprocessed_test_tweets * test_weight))

    def all_tokenized_tweets(self):
        tokenized_tweets_randomized  = [self.preprocessor.lexical_preprocessing_tweet(tweet)
                                        for tweet in self.shuffled_original_train_tweets]
        tokenized_tweets_randomized += [self.preprocessor.lexical_preprocessing_tweet(tweet)
                                        for tweet in self.original_test_tweets]
        #random.shuffle(tokenized_tweets_randomized)
        tokenized_tweets_randomized = numpy_random_shuffle(tokenized_tweets_randomized)
        return tokenized_tweets_randomized

    def shuffle_and_split(self, input_names=None):
        """Create training data set"""
        print("[TrainingDataset] Shuffling data %s..."
              % ('for inputs [' + ', '.join(input_names) + ']') if input_names is not None else '')
        combined_training_dataset = list(
            zip(self.shuffled_original_train_tweets, self.shuffled_train_tweets, self.shuffled_train_sentiments))
        #random.shuffle(combined_training_dataset)
        combined_training_dataset = numpy_random_shuffle(combined_training_dataset)
        self.shuffled_original_train_tweets[:], self.shuffled_train_tweets[:], self.shuffled_train_sentiments[:] = zip(
            *combined_training_dataset)

        nb_validation_samples = int(self.training_validation_split_ratio * len(self.shuffled_train_tweets))

        x_train = self.shuffled_train_tweets[:nb_validation_samples]
        x_orig_train = self.shuffled_original_train_tweets[:nb_validation_samples]
        y_train = self.shuffled_train_sentiments[:nb_validation_samples]
        x_val = self.shuffled_train_tweets[nb_validation_samples:]
        x_orig_val = self.shuffled_original_train_tweets[nb_validation_samples:]
        y_val = self.shuffled_train_sentiments[nb_validation_samples:]

        x_train = tweets_to_inputs(x_train,input_names)
        x_val   = tweets_to_inputs(x_val,input_names)

        return (x_train, y_train, x_orig_train), (x_val, y_val, x_orig_val)

    def shuffle_and_split_padded(self, input_names=None):
        (x_train, y_train, x_orig_train), (x_val, y_val, x_orig_val) = self.shuffle_and_split(input_names)
        return (self.pad_tweets(x_train), y_train, x_orig_train), (self.pad_tweets(x_val), y_val, x_orig_val)

    def test_tweets(self, input_names=None):
        print("[TrainingDataset] Generating test data %s..."
              % ('for inputs [' + ', '.join(input_names) + ']') if input_names is not None else '')
        return tweets_to_inputs(self._test_tweets,input_names)

    def test_tweets_padded(self, input_names=None):
        return self.pad_tweets(self.test_tweets(input_names))

    def pad_tweets(self,tweets):
        if isinstance(tweets,dict):
            padded_tweets = {}
            for name in tweets:
                padded_tweets[name] = sequence.pad_sequences(tweets[name],
                                                             maxlen=self.max_tweet_length,
                                                             value=self.preprocessor.vocabulary.word_to_id['<pad>'])
        else:
            padded_tweets = sequence.pad_sequences(tweets,
                                                   maxlen=self.max_tweet_length,
                                                   value=self.preprocessor.vocabulary.word_to_id['<pad>'])
        return padded_tweets

def reverse_tweets(tweets):
    return [tweet[::-1] for tweet in tweets]

def tweets_to_inputs(tweets,input_names):
    if (input_names is not None
        and len(input_names) == 2
        and input_names[0] == 'forward_input'
        and input_names[1] == 'backward_input'):
        inputs = {'forward_input': tweets,
                  'backward_input': reverse_tweets(tweets)}
    else:
        inputs = tweets
    return inputs

def pad_tweets(tweets, max_tweet_length, preprocessor):
    if isinstance(tweets, dict):
        padded_tweets = {}
        for name in tweets:
            padded_tweets[name] = sequence.pad_sequences(tweets[name],
                                                         maxlen=max_tweet_length,
                                                         value=preprocessor.vocabulary.word_to_id['<pad>'])
    else:
        padded_tweets = sequence.pad_sequences(tweets,
                                               maxlen=max_tweet_length,
                                               value=preprocessor.vocabulary.word_to_id['<pad>'])
    return padded_tweets

def unpad_tweets(tweets, preprocessor):
    if isinstance(tweets, dict):
        unpadded_tweets = {}
        for name in tweets:
            unpadded_tweets[name] = sequence.pad_sequences(tweets[name],
                                                         maxlen=max_tweet_length,
                                                         value=preprocessor.vocabulary.word_to_id['<pad>'])
    else:
        unpadded_tweets = [unpad_tweet(tweet) for tweet in tweets]
    return unpadded_tweets

def unpad_tweet(tweet, preprocessor):
    return [word for word in filter(lambda w: w != preprocessor.preprocessor.vocabulary['<pad>'],tweet)]
