import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
files = os.listdir(args.input_folder)

#####


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
num_epochs = 5
batch_size = 70
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([256, 256]),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615))])
test_dataset = CustomImageDataset(args.input_folder, transform=transform)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
cnn = CNN()
cnn.load_state_dict(torch.load('./final_model.pkl'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn = cnn.cuda(device)
cnn.eval()

prediction_df = pd.DataFrame(columns=['name', 'predicted'])
for i, (images, labels, names) in enumerate(test_loader):
    labels = torch.tensor([int(i) for i in labels])
    if torch.cuda.is_available():
        images = images.cuda()

    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.tolist()
    for j in range(len(labels)):
        new_row = pd.DataFrame(data={'name': f'{names[j]}_{labels[j]}.jpg', 'predicted': int(predicted[j])},
                               index={j})
        prediction_df = pd.concat([prediction_df, new_row])


prediction_df.to_csv("prediction.csv", index=False, header=False)
