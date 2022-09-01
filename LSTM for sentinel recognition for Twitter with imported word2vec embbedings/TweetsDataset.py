import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import gensim.utils as ut
from gensim.models import KeyedVectors, Word2Vec


class TweetsDataset(Dataset):
    def __init__(self, df, wv, max_len):
        self.df = df
        self.wv = wv
        self.labels = list(df.emotion)
        self.max_len = max_len

    def get_vocab(self):
        vocab = []
        prog = re.compile("\w+")
        vocab_list = list(self.df['content'])
        for line in vocab_list:
            vocab.append(prog.findall(line))
        return vocab

    def update_wv(self, new_wv):
        self.wv = new_wv


    def turn2embedding(self, token_tweet):
        token = self.wv[token_tweet]
        if token.shape[0] > self.max_len:
            return token[:self.max_len]
        return torch.from_numpy(np.pad(token, ((0, self.max_len - token.shape[0]), (0, 0)), mode='mean'))

    def label_to_class(self, label):
        if label == 'happiness':
            return 2
        if label == 'neutral':
            return 1
        if label == 'sadness':
            return 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        prog = re.compile("\w+")
        tweet, label = self.df.content.iloc[idx], self.labels[idx]
        label = self.label_to_class(label)
        token_tweet = prog.findall(tweet)
        return self.turn2embedding(token_tweet), tweet, label
