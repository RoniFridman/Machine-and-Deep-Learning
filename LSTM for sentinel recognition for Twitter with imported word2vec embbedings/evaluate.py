import sys
import argparse
import numpy as np
import pandas as pd
import torch
from TweetsDataset import TweetsDataset
from LSTM import LSTM
from update_word2vec import update_word2vec
from gensim.models import KeyedVectors, Word2Vec
from sklearn.metrics import accuracy_score


def translate_label(row):
    label = row['emotion']
    if label == 0:
        row['emotion'] = 'sadness'
    if label == 1:
        row['emotion'] = 'neutral'
    if label == 2:
        row['emotion'] = 'happiness'
    return row


def evaluate(test_path, model_path='model.pkl'):
    with torch.no_grad():
        # Command line: $ python predict.py path/testEmotions.csv
        num_classes = 3
        input_size = 100
        hidden_size = 128
        num_layers = 2
        max_len = 131
        test_df = pd.read_csv(test_path, header=0)
        test_set = TweetsDataset(test_df, None, max_len)
        test_vocab = test_set.get_vocab()
        update_word2vec(test_vocab, no_header=False)
        w2v_model = KeyedVectors.load_word2vec_format("model_vocab.txt", no_header=False, binary=False)
        test_set.update_wv(w2v_model)
        test_loader = torch.utils.data.dataloader.DataLoader(test_set)
        print('starting evaluation')
        lstm = LSTM(num_classes, input_size, hidden_size, num_layers, max_len)
        lstm.load_state_dict(torch.load(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lstm = lstm.cuda(device)
        lstm.eval()
        train_loss_per_epoch = {}
        prediction_df = pd.DataFrame(columns=['label', 'emotion', 'content'])
        for i, (tensors, tweets, labels) in enumerate(test_loader):
            labels = torch.tensor([int(i) for i in labels])
            if torch.cuda.is_available():
                tensors = tensors.cuda()
                labels = labels.cuda()

            outputs = lstm(tensors)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.tolist()
            for j in range(len(labels)):
                new_row = pd.DataFrame(data={'label': int(labels[j]), 'emotion': int(predicted[j]), 'content': tweets[j]},
                                       index={j})
                prediction_df = pd.concat([prediction_df, new_row])
            train_loss_per_epoch[i] = accuracy_score(list(prediction_df['label']), list(prediction_df['emotion']))

        print(f" Accuracy: {accuracy_score(list(prediction_df['label']), list(prediction_df['emotion']))}")
        prediction_df = prediction_df.drop(columns=['label'])
        prediction_df.apply(translate_label, axis=1)
        prediction_df.to_csv('predictions.csv', index=False)
        test_loss_df = pd.DataFrame.from_dict(train_loss_per_epoch, orient='index')
        test_loss_df.to_csv('test_loss_dict.csv', header=False)

