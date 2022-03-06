import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings('ignore')
from vmdpy import VMD
from scipy import stats
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

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

def _VMD(tot_dfs, lengths, partitions):
    for select in range(1, 251):
        tot_select = tot_dfs[select]
        length_select = lengths[select]
        partition_select = partitions[select]

        # moderate bandwidth constraint
        alpha = length_select * 2
        # noise-tolerance (no strict fidelity enforcement)
        tau = 0.
        # 3 modes
        K = 3
        # no DC part imposed
        DC = 0
        # initialize omegas uniformly
        init = 1
        tol = 1e-7

        u, u_hat, omega = VMD(tot_select, alpha, tau, K, DC, init, tol)
        kurtosis = stats.kurtosis(omega[:, -1])

        while kurtosis < 0 and K < 10:
            K = K + 1
            u, u_hat, omega = VMD(tot_select, alpha, tau, K, DC, init, tol)
            kurtosis = stats.kurtosis(omega[:, -1])

        while K < 10:
            K = K + 1
            u_new, u_hat_new, omega_new = VMD(tot_select, alpha, tau, K, DC, init, tol)
            kurtosis_new = stats.kurtosis(omega_new[:, -1])

            if kurtosis_new >= kurtosis:
                u = u_new
                u_hat = u_hat_new
                omega = omega_new
                kurtosis = kurtosis_new
            else:
                break

        K = K - 1


        np.savetxt(f'C:/Users/msi-/Desktop/21Fall BDT/Data Mining/project/data-sets/data-sets/KDD-Cup/VMD/DC=0_file/No.{select}_VMD.txt', u[0, :], fmt='%.2f')


        plt.rcParams['figure.figsize'] = (30, 30)
        plt.suptitle(f'No.{select} VMD', x=0.51, y=0.92)

        for n, imf in enumerate(u):
            plt.subplot(u.shape[0] + 1, 1, n + 1)
            plt.plot(np.arange(imf.shape[0]), imf)
            plt.axvline(partition_select, color='r', linestyle='dashed')
            plt.title("IMF " + str(n + 1))

        plt.subplots_adjust(wspace = 0, hspace = 0.3)
        plt.savefig(
            f'C:/Users/msi-/Desktop/21Fall BDT/Data Mining/project/data-sets/data-sets/KDD-Cup/VMD/DC=0/No.{select}_VMD.png',
            dpi=1000, bbox_inches='tight')
        plt.close()

        print(f"{select} done")

def main():
    data_path = "data-sets/KDD-Cup/data"
    tot_dfs = dataLoad(data_path)

    lengths = np.loadtxt("data-sets/KDD-Cup/lengths.txt", dtype='int32')
    partitions = np.loadtxt("data-sets/KDD-Cup/partitions.txt", dtype='int32')

    _VMD(tot_dfs, lengths, partitions)

if __name__ == "__main__":
    main()