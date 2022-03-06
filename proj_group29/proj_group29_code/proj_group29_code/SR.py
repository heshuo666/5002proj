import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import math
from matrixprofile import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import sranodec as anom

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

def SR(tot_dfs, periods):
    for select in range(1, 251):
        window_size = periods[select]
        dfs_select = tot_dfs[select]

        amp_window_size = int(window_size / 2)
        series_window_size = int(window_size / 2)
        score_window_size = int(window_size * 2)

        spec = anom.Silency(amp_window_size, series_window_size, score_window_size)
        score = spec.generate_anomaly_score(dfs_select)

        np.savetxt(f"data-sets/KDD-Cup/SR/{select}_SR.txt", score, fmt='%.06f')

        plt.plot(score, color='g')
        plt.title(f'{select}_SR')
        plt.savefig(
            f"data-sets/KDD-Cup/SR/figures/{select}_SR_figure.png",
            dpi=1000, bbox_inches='tight')
        plt.close()

        print(f"{select} done")



def main():
    data_path = "data-sets/KDD-Cup/data"
    tot_dfs = dataLoad(data_path)
    periods = np.loadtxt("data-sets/KDD-Cup/period_list.txt", dtype='int32')

    SR(tot_dfs, periods)

if __name__ == "__main__":
    main()