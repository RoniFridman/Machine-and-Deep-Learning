import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = {}
        idx = 0
        for img in os.listdir(img_dir):
            ID, label = str.split(img, '_')
            self.img_labels[idx] = (ID, label)
            idx += 1

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = f"{self.img_labels[idx][0]}_{self.img_labels[idx][1]}"
        name = f"{self.img_labels[idx][0]}"
        img_path = os.path.join(self.img_dir, img_path)
        image = Image.open(img_path)

        label = self.img_labels[idx][1][0]
        if self.transform:
            image = self.transform(image)
        return image, label, name


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(256 * 256 * 2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return self.logsoftmax(out)


# Hyper Parameters
num_epochs = 5
batch_size = 70
learning_rate = 0.001

train_path = '/home/student/HW1/data/raw_data/train/train'
test_path = '/home/student/HW1/data/raw_data/test/test'
train_dataset = CustomImageDataset(train_path, transform=transform)
test_dataset = CustomImageDataset(test_path, transform=transform)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

cnn = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn = cnn.cuda(device)

# convert all the weights tensors to cuda()
# Loss and Optimizer

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
train_loss_per_epoch = {}
test_loss_per_epoch = {}
F1_per_epoch = {}
ROC = []
for epoch in range(num_epochs):
    train_loss = []
    test_loss = []
    F1 = []
    for i, (images, labels, names) in enumerate(train_loader):
        labels = torch.tensor([int(j) for j in labels])
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        # Forward + Backward + Optimize
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        train_loss.append(loss.item())
        output_array = outputs.cpu().detach().numpy()
        labels_array = labels.cpu().detach().numpy()
        TP, TN, FN, FP = (0, 0, 0, 0)
        for i in range(len(labels_array)):
            y_pred = np.argmax(output_array[i])
            if y_pred == labels_array[i]:
                if labels_array[i] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if labels_array[i] == 1:
                    FP += 1
                else:
                    FN += 1
        ROC.append((FP / (TN + FP), TP / (TP + FN)) if TP != 0 and TN != 0 else (0, 0))
        F1.append(2 * TP / (2 * TP + FP + FN))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1,
                     len(train_dataset) // batch_size, loss.data))
    train_loss_per_epoch[epoch] = train_loss
    F1_per_epoch[epoch] = F1
    for i, (images1, labels1, names1) in enumerate(test_loader):
        labels1 = torch.tensor([int(j) for j in labels1])
        images1 = images1.cuda()
        labels1 = labels1.cuda()
        outputs = cnn(images1)
        loss = criterion(outputs, labels1)
        test_loss.append(loss.item())
    test_loss_per_epoch[epoch] = test_loss

with open('ROC_train_list.txt', 'w') as f:
    for i in ROC:
        f.write(str(i))
f.close()

torch.save(cnn.state_dict(), 'cnn_256_3layers_FINAL.pkl')
df = pd.DataFrame.from_dict(train_loss_per_epoch, orient='index')
df.to_csv('train_loss_dict.csv', header=False)

df = pd.DataFrame.from_dict(test_loss_per_epoch, orient='index')
df.to_csv('test_loss_dict.csv', header=False)

df = pd.DataFrame.from_dict(F1_per_epoch, orient='index')
df.to_csv('F1_score.csv', header=False)
