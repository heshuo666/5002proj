import numpy as np
import pandas as pd
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import sys
import os

def dataLoad(data_path):
    dir_list = os.listdir(data_path)
    tot_dfs = [None]

    anomaly_file_1 = set([204, 205])
    anomaly_file_2 = set([206, 207, 208, 242, 243, 225, 226])
    index = 1

    for filename in dir_list[0: 250]:
        df = pd.read_csv(data_path + '/' + filename, names=['values'])

        if index in anomaly_file_1:
            newDataFrame = pd.DataFrame()
            newDataFrame['values'] = list(map(float, list(df.iloc[0])[0].strip(' ').split('   ')))
            df = newDataFrame
        elif index in anomaly_file_2:
            newDataFrame = pd.DataFrame()
            newDataFrame['values'] = list(map(float, list(df.iloc[0])[0].strip(' ').split('  ')))
            df = newDataFrame

        df = df.values.ravel()
        tot_dfs.append(df)
        index += 1

    return tot_dfs

def preprocessing(dfs_select, window_size, partition):
    dfs_train = dfs_select[: partition]
    dfs_test = dfs_select[partition:]
    dfs_test_length = dfs_test.shape[0]
    train_window_samples = []
    test_window_samples = []
    for index in range(0, partition):
        if index + window_size > partition:
            break
        train_window_sample = dfs_train[index: index + window_size]
        train_window_samples.append(train_window_sample)
    train_window_samples = np.array(train_window_samples)
    for index in range(0, dfs_test_length):
        if index + window_size > dfs_test_length:
            break
        test_window_sample = dfs_test[index: index + window_size]
        test_window_samples.append(test_window_sample)
    test_window_samples = np.array(test_window_samples)

    return train_window_samples, test_window_samples

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, height, wide):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, (4, height))
        self.conv2 = nn.Conv1d(8, 16, (4, 1))
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv1d(16, 32, (4, 1))
        self.linear1 = nn.Linear(32 * (wide - 9) * 1, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (0.5 * sigma + 0.5 * mu ** 2 - 0.5 * torch.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims, height, wide):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 32 * (wide - 27) * height),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, wide - 27, int(height)))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, (10, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, (10, 1)),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, (10, 1))
        )
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, height, wide):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, height, wide)
        self.decoder = Decoder(latent_dims, height, wide)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_epoch(vae, device, train_window_samples, optimizer):
    vae.train()
    train_loss = 0.0
    for x in train_window_samples:
        x = x[np.newaxis, np.newaxis, ]
        x = torch.from_numpy(x).float().to(device)
        x_hat = vae(x)
        loss = ((x - x_hat)**2).sum() + vae.encoder.kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / (train_window_samples.shape[0] * train_window_samples.shape[1] * train_window_samples.shape[2])

def _test_epoch(vae, device, test_window_samples):
    vae.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x in test_window_samples:
            x = x[np.newaxis, np.newaxis, ]
            x = torch.from_numpy(x).float().to(device)
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            test_loss += loss.item()
    return test_loss / (test_window_samples.shape[0] * test_window_samples.shape[1] * test_window_samples.shape[2])

def main():
    data_path = "data-sets/KDD-Cup/data"
    tot_dfs = dataLoad(data_path)
    periods = np.loadtxt("data-sets/KDD-Cup/period_list.txt", dtype='int32')
    partitions = np.loadtxt("data-sets/KDD-Cup/partitions.txt", dtype='int32')

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f'Selected device: {device}')

    for select in range(1, 2):
        window_size = periods[select]
        dfs_select = tot_dfs[select]
        partition = partitions[select]
        train_window_samples, test_window_samples = preprocessing(dfs_select, window_size, partition)

        wide = 150

        train_window_samples = train_window_samples[: train_window_samples.shape[0] - train_window_samples.shape[0] % wide, :]
        test_window_samples = test_window_samples[: test_window_samples.shape[0] - test_window_samples.shape[0] % wide, :]

        print(train_window_samples.shape)
        print(test_window_samples.shape)

        train_window_samples.resize(int(train_window_samples.size / (wide * window_size)), wide, window_size)
        test_window_samples.resize(int(test_window_samples.size / (wide * window_size)), wide, window_size)

        seed = 666
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        """
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        print(f'Selected device: {device}')
        """

        vae = VariationalAutoencoder(latent_dims=64, height=window_size, wide=wide)
        vae.to(device)
        lr = 1e-3 * 0.5
        optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

        vae.to(device)

        num_epochs = 1000
        for epoch in range(num_epochs):
            train_loss = train_epoch(vae, device, train_window_samples, optimizer)
            test_loss = _test_epoch(vae, device, test_window_samples)
            print(f'EPOCH {epoch + 1}/{num_epochs} train loss {train_loss} test loss {test_loss}')



if __name__ == "__main__":
    main()