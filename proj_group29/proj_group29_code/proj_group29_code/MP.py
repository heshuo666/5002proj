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

def MP(tot_dfs, periods):
    for select in range(219, 251):
        window_size = periods[select]
        dfs_select = tot_dfs[select]
        MP_result = matrixProfile.scrimp_plus_plus(dfs_select, window_size)

        np.savetxt(f"data-sets/KDD-Cup/MP/{select}_MP0.txt", MP_result[0], fmt='%.06f')
        np.savetxt(f"data-sets/KDD-Cup/MP/{select}_MP1.txt", MP_result[1], fmt='%d')

        plt.figure()
        mpl.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams['figure.figsize'] = (20, 16)
        plt.title(f"{select}_MP0")
        plt.plot(MP_result[0], linewidth=0.5)
        plt.savefig(
            f"data-sets/KDD-Cup/MP/figures/{select}_MP0_figure.png",
            dpi=1000, bbox_inches='tight')
        plt.close()

        print(f"{select} done")


def main():
    data_path = "data-sets/KDD-Cup/data"
    tot_dfs = dataLoad(data_path)
    periods = np.loadtxt("data-sets/KDD-Cup/period_list.txt", dtype='int32')

    MP(tot_dfs, periods)

if __name__ == "__main__":
    main()