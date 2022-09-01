import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, max_len):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,dropout=0.005)  # lstm
        self.fc_1 = nn.Linear(hidden_size*self.max_len, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, line):
        batch_size = line.shape[0]
        h0 = torch.zeros(2, batch_size, 128)
        c0 = torch.zeros(2, batch_size, 128)
        h0 = h0.cuda()
        c0 = c0.cuda()
        output, (hn, cn) = self.lstm(line, (h0, c0))
        output = output.reshape(batch_size, self.max_len*128)
        out = self.relu(output)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        out = nn.LogSoftmax(dim=1)(out)
        return out

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
