import numpy as np
# np.set_printoptions(threshold=np.inf)
import pandas as pd
import random
import torch
# torch.set_printoptions(threshold=np.inf)
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary
import sys
import os

from matrixprofile import *
import matplotlib.pyplot as plt
import matplotlib as mpl

import warnings
warnings.filterwarnings('ignore')

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
        train_window_sample_0 = np.array(dfs_train[index: index + window_size])
        train_window_samples.append((train_window_sample_0, train_window_sample_0))
    for index in range(0, dfs_test_length):
        if index + window_size > dfs_test_length:
            break
        test_window_sample_0 = np.array(dfs_test[index: index + window_size])
        test_window_samples.append((test_window_sample_0, test_window_sample_0))

    test_window_samples_final = []

    for index in range(0, dfs_test_length, window_size):
        if index + window_size > dfs_test_length:
            break
        test_window_sample_final_0 = np.array(dfs_test[index: index + window_size])
        test_window_samples_final.append(test_window_sample_final_0)

    return train_window_samples, test_window_samples, test_window_samples_final

class LSTMpred(nn.Module):
    def __init__(self, input_size, hidden_dim, device):
        super(LSTMpred, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim).to(self.device), torch.zeros(1, 1, self.hidden_dim).to(self.device))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        out_date = self.hidden2out(lstm_out.view(len(seq), -1))
        return out_date

def train_epoch(model, device, train_window_samples, optimizer, loss_function, epoch, num_epochs):
    model.train()
    train_loss = 0.0
    flag = 0
    for seq, out in train_window_samples:
        seq = torch.from_numpy(seq).float().to(device)
        out = torch.from_numpy(out).float().to(device)

        optimizer.zero_grad()

        model.hidden = model.init_hidden()

        model_out = model(seq)

        loss = loss_function(model_out, out)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if flag == 0 and (epoch % 5 == 0 or epoch == num_epochs - 1):
            mpl.rcParams['agg.path.chunksize'] = 10000
            plt.rcParams['figure.figsize'] = (20, 16)
            plt.plot(range(seq.shape[0]), seq.cpu().detach().numpy(), color='green', label="seq")
            plt.plot(range(1, seq.shape[0] + 1), model_out.cpu().detach().numpy(), color='red', label="model_out")
            plt.title('1' + ' train sample ' + f'epoch {epoch}')
            plt.legend()
            plt.savefig(
                f'data-sets/KDD-Cup/Regression_LSTM/{epoch}_train_sample.png',
                dpi=1000, bbox_inches='tight')
            plt.close()

            print(f"train {epoch + 1} flag done")

        flag += 1

    return train_loss / (len(train_window_samples) * train_window_samples[0][0].shape[0])

def _test_epoch(model, device, test_window_samples, loss_function, epoch, num_epochs):
    # print("hello")
    model.eval()
    test_loss = 0.0
    flag = 0
    for seq, out in test_window_samples:
        seq = torch.from_numpy(seq).float().to(device)
        out = torch.from_numpy(out).float().to(device)

        model.hidden = model.init_hidden()

        model_out = model(seq)

        loss = loss_function(model_out, out)

        test_loss += loss.item()

        if flag == 0 and (epoch % 5 == 0 or epoch == num_epochs - 1):
            mpl.rcParams['agg.path.chunksize'] = 10000
            plt.rcParams['figure.figsize'] = (20, 16)
            plt.plot(range(seq.shape[0]), seq.cpu().detach().numpy(), color='green', label="seq")
            plt.plot(range(1, seq.shape[0] + 1), model_out.cpu().detach().numpy(), color='red', label="model_out")
            plt.title('1' + ' test sample ' + f'epoch {epoch}')
            plt.legend()
            plt.savefig(
                f'data-sets/KDD-Cup/Regression_LSTM/{epoch}_test_sample.png',
                dpi=1000, bbox_inches='tight')
            plt.close()

            print(f"test {epoch + 1} flag done")

        flag += 1

    return test_loss / (len(test_window_samples) * test_window_samples[0][0].shape[0])




def Regeression_LSTM(tot_dfs, periods, partitions):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f'Selected device: {device}')

    for select in range(1, 251):

        window_size = periods[select]
        dfs_select = tot_dfs[select]
        partition = partitions[select]
        train_window_samples, test_window_samples, test_window_samples_final = preprocessing(dfs_select, window_size, partition)

        seed = 666
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = LSTMpred(1, 6, device)
        model.to(device)
        loss = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        num_epochs = 5

        for epoch in range(num_epochs):
            print(f"epoch {epoch + 1} starts!")
            train_loss = train_epoch(model, device, train_window_samples, optimizer, loss, epoch, num_epochs)
            test_loss = _test_epoch(model, device, test_window_samples, loss, epoch, num_epochs)
            print(f'EPOCH {epoch + 1}/{num_epochs} train loss {train_loss} test loss {test_loss}')

        test_final = []
        model.eval()
        for seq in test_window_samples_final:
            seq = torch.from_numpy(seq).float().to(device)
            model.hidden = model.init_hidden()
            model_out = model(seq)
            test_final += model_out.cpu().detach().numpy().tolist()

        test_final = np.array(test_final).flatten()

        np.savetxt(f"data-sets/KDD-Cup/Regression_LSTM/{select}_RL.txt", test_final, fmt='%.06f')

        mpl.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams['figure.figsize'] = (20, 16)
        plt.plot(dfs_select[partition:], color='green', label="seq", linewidth=0.5)
        plt.plot(test_final, color='red', label="model_out", linewidth=0.5)
        plt.title(f'{select}_test_final')
        plt.legend()
        plt.savefig(
            f'data-sets/KDD-Cup/Regression_LSTM/{select}_test_final.png',
            dpi=1000, bbox_inches='tight')
        plt.close()



def main():
    data_path = "data-sets/KDD-Cup/data"
    tot_dfs = dataLoad(data_path)

    periods = np.loadtxt("data-sets/KDD-Cup/period_list.txt", dtype='int32')
    partitions = np.loadtxt("data-sets/KDD-Cup/partitions.txt", dtype='int32')

    Regeression_LSTM(tot_dfs, periods, partitions)

if __name__ == "__main__":
    main()