import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import string
import re
from torch.utils.data import DataLoader
from TweetsDataset import TweetsDataset
from LSTM import LSTM
import wget
import bz2
import zipfile
from update_word2vec import update_word2vec
from gensim.models import KeyedVectors, Word2Vec
# from gensim.test.utils import datapath, get_tmpfile
# from gensim.scripts.glove2word2vec import glove2word2vec
from smart_open import compression
from evaluate import evaluate


def train():
    """
    Train the model
    :param lr: lerning rate of optimizer
    :param epochs: epochs of training
    :return: saves a pkl file of the model.
    """

    # download embeddings and create a txt file of them
    wget.download('https://nlp.stanford.edu/data/glove.6B.zip')
    with zipfile.ZipFile('glove.6B.zip', 'r') as zip_ref:
        zip_ref.extractall("./")
    with open('glove.6B.100d.txt', 'r') as f, open('model_vocab.txt', 'w') as t:
        for line in f:
            t.write(f'{line}')

    num_epochs = 8
    learning_rate = 0.002
    num_classes = 3
    input_size = 100
    hidden_size = 128
    num_layers = 2
    max_len = 131
    batch_size = 100

    # create training set
    train_df = pd.read_csv('trainEmotions.csv', header=0)
    train_set = TweetsDataset(train_df, None, max_len)
    train_vocab = train_set.get_vocab()
    update_word2vec(train_vocab)
    w2v_model = KeyedVectors.load_word2vec_format("model_vocab.txt", no_header=False, binary=False)
    train_set.update_wv(w2v_model)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True)

    # run training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.NLLLoss()
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, max_len)
    lstm = lstm.cuda(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    train_loss_per_epoch = {}
    print('start training')
    for epoch in range(num_epochs):
        train_loss = []
        for i, (tensors, tweets, labels) in enumerate(train_loader):
            labels = torch.tensor([j for j in labels])
            if torch.cuda.is_available():
                tensors = tensors.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            output = lstm(tensors)
            predicted = output.to(device)
            loss = criterion(predicted, labels.type(torch.LongTensor).to(device))
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        train_loss_per_epoch[epoch] = train_loss

    df = pd.DataFrame.from_dict(train_loss_per_epoch, orient='index')
    df.to_csv('train_loss_dict.csv', header=False)
    torch.save(lstm.state_dict(), 'model.pkl')
