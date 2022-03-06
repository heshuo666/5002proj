import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')
from math import *
from matrixprofile import *
import matplotlib.pyplot as plt
import matplotlib as mpl

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

def sliding_window(tot_dfs, periods):
    for select in range(1, 251):
        window_size = periods[select]
        dfs_select = tot_dfs[select]

        weight = np.ones(window_size) / window_size
        sliding_window_result = np.convolve(dfs_select, weight, mode='valid')
        diff = sliding_window_result - dfs_select[: sliding_window_result.shape[0]]


        np.savetxt(f"data-sets/KDD-Cup/sliding_window/{select}_SW_diff.txt", diff, fmt='%.06f')

        plt.title(f"{select}_SW_diff")
        plt.plot(diff, linewidth=0.5)
        plt.savefig(f"data-sets/KDD-Cup/sliding_window/figures/{select}_SW_diff.png", dpi=1000, bbox_inches='tight')
        plt.close()

        print(f"{select} done")


def main():
    data_path = "data-sets/KDD-Cup/data"
    tot_dfs = dataLoad(data_path)

    periods = np.loadtxt("data-sets/KDD-Cup/period_list.txt", dtype='int32')

    sliding_window(tot_dfs, periods)

if __name__ == "__main__":
    main()