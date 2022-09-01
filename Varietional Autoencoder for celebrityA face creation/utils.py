import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_train_loader(argv, train_dataset, batch_size):
    """ Initialize train DataLoader based on a command line argument.
    :param argv: list of command line inputs
    :param train_dataset: CelebA dataset
    :param batch_size: batch size
    :return: DataLoader
    """
    train_loader = None
    if len(argv) > 2:
        if argv[2] == 'save':
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True)
            torch.save(train_loader, 'dataloader.pt')
        elif argv[2] == 'load':
            train_loader = torch.load('dataloader.pt')
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    return train_loader


def train_disc(model, data_loader, optimizer, annel_rate, temp_min, epochs, temp, hard):
    print('Training discrete model')
    agg_loss = []
    model.train()
    bce_l_total = 0
    kld_total = 0
    for epoch in tqdm(range(epochs)):
        agg_loss.append((bce_l_total, kld_total, epoch))
        bce_l_total = 0
        kld_total = 0
        for batch_idx, (x, _) in tqdm(enumerate(data_loader)):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, qy = model(x, temp, hard)
            bce, kld = model.loss_function(x_hat, x, qy)
            bce_l_total = bce_l_total + bce.item()
            kld_total = kld_total + kld.item()
            loss = bce + kld
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 1:
                temp = np.maximum(temp * np.exp(-annel_rate * batch_idx), temp_min)
            # if batch_idx % 100 == 0:
            #     print(batch_idx)
    return agg_loss


def train_cont(model, data_loader, opt, epochs):
    print('Training continues model')
    agg_loss = []
    bce_l_total = 0
    kld_total = 0
    model.train()
    for epoch in tqdm(range(epochs)):
        agg_loss.append((bce_l_total, kld_total, epoch))
        bce_l_total = 0
        kld_total = 0
        for batch_idx, (x, _) in tqdm(enumerate(data_loader)):
            x = x.to(device)
            opt.zero_grad()
            x_hat = model(x)
            # compute BCE and KLD separately to showcase them by time
            bce_l = F.binary_cross_entropy(x_hat, x, reduction='sum')
            kld = model.encoder.kl
            loss = bce_l + kld
            loss.backward()
            bce_l_total = bce_l_total + bce_l.item()
            kld_total = kld_total + kld.item()
            opt.step()
            # if batch_idx % 100 == 0:
            #     print(batch_idx)
    return agg_loss
