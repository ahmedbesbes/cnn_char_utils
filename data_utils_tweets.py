# -*- coding: utf-8 -*-

import numpy as np
import re
import csv
import pandas as pd
from tqdm import tqdm


class Data(object):

    """
    Class to handle loading and processing of raw datasets.
    """
    def __init__(self, data_source,
                 alphabet="abcdefghijklmnopqrstuvwxyzéàèùâêîôûçëïü0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                 alphabet_size=82,
                 input_size=1014, num_of_classes=4, max_len_sentence=350, n_tweets=None):
        """
        Initialization of a Data object.

        Args:
            data_source (str): Raw data file path
            alphabet (str): Alphabet of characters to index
            input_size (int): Size of input features
            num_of_classes (int): Number of classes in data
        """
        self.alphabet = alphabet
        self.alphabet_size = alphabet_size
        self.dict = {}  # Maps each character to an integer
        self.no_of_classes = num_of_classes
        for idx, char in enumerate(self.alphabet):
            self.dict[char] = idx + 1
        self.length = input_size
        self.data_source = data_source
        self.max_len_sentence = max_len_sentence
        self.n_tweets = n_tweets

    def load_data(self):
        """
        Load raw data from the source file into data variable.

        Returns: None

        """
        data = []
        
        if self.n_tweets is not None:
            aux = pd.read_csv(self.data_source, encoding='latin1', usecols=['sentiment', 'cleaned_text'], nrows=self.n_tweets)
        else:
            aux = pd.read_csv(self.data_source, encoding='latin1', usecols=['sentiment', 'cleaned_text'])
            
        print('data shape', aux.shape, 'with columns :', list(aux.columns))

        for review, label in zip(aux['sentiment'], aux['cleaned_text']):
            data.append((label, review))
            
        self.data = np.array(data)
        print("Data loaded from " + self.data_source)

    def get_all_data(self):
        """
        Return all loaded data from data variable.

        Returns:
            (np.ndarray) Data transformed from raw to indexed form with associated one-hot label.

        """
        data_size = len(self.data)
        start_index = 0
        end_index = data_size
        batch_texts = self.data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in tqdm(batch_texts):
            batch_indices.append(self.str_to_indexes(c))
            # c = int(c) - 1
            classes.append(int(s))
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

    def str_to_indexes(self, s):
        """
        Convert a string to character indexes based on character dictionary.
        
        Args:
            s (str): String to be converted to indexes

        Returns:
            str2idx (np.ndarray): Indexes of characters in s

        """
        s = s.lower()
        max_length = min(len(s), self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        for i in range(1, max_length + 1):
            c = s[-i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]
        return str2idx
