import os
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


def evaluate(loader='real_test_loader.pth', model_path='cnn_256_3layers_FINAL.pkl'):
    test_loader = torch.load(loader)
    cnn = CNN()
    cnn.load_state_dict(torch.load(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = cnn.cuda(device)
    cnn.eval()
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    ROC = []

    prediction_df = pd.DataFrame(columns=['id', 'label', 'predicted'])
    for i, (images, labels, names) in enumerate(test_loader):
        labels = torch.tensor([int(i) for i in labels])
        if torch.cuda.is_available():
            images = images.cuda()

        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.tolist()
        for j in range(len(labels)):
            new_row = pd.DataFrame(data={'id': names[j], 'label': int(labels[j]), 'predicted': int(predicted[j])},
                                   index={j})
            prediction_df = pd.concat([prediction_df, new_row])
            if predicted[j] == labels[j]:
                if predicted[j] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if predicted[j] == 1:
                    FP += 1
                else:
                    FN += 1
            ROC.append((TP/(TP+FN), FP/(TN+FP)) if TP !=0 and TN != 0 else (0,0))

    print(f'F1 Measure is:{2 * TP / (2 * TP + FP + FN)}')
    with open('ROC_test_list.txt', 'w') as f:
        for i in ROC:
            f.write(str(i))
    f.close()


def main():
    with torch.no_grad():
        evaluate()


if __name__ == "__main__":
    main()
