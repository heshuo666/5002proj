import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import math

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

def auto_period_finder(tot_dfs, f):
    period_list = [0]
    for select in range(1, 251):
        dfs_select = tot_dfs[select]
        dfs_select_length = len(dfs_select)
        max_step = int(dfs_select_length / 250)
        scores = [float('inf')]
        for step in range(1, max_step + 1):
            peaks = []
            valleys = []

            for index in range(0, dfs_select_length, step):
                if index + step <= dfs_select_length:
                    peak = np.argmax(dfs_select[index: index + step]) + index
                    valley = np.argmin(dfs_select[index: index + step]) + index
                    peaks.append(peak)
                    valleys.append(valley)

            peaks = np.array(peaks)
            valleys = np.array(valleys)
            score = min(np.std(peaks), np.std(valleys)) / math.sqrt(step)
            scores.append(score)

        scores = np.array(scores)
        print(select, np.argmin(scores))
        f.write(str(np.argmin(scores)) + '\n')
        period_list.append(np.argmin(scores))

    return period_list

def main():
    data_path = "data-sets/KDD-Cup/data"
    tot_dfs = dataLoad(data_path)
    f = open("data-sets/KDD-Cup/period_list.txt", "w")
    period_list = auto_period_finder(tot_dfs, f)
    f.close()

if __name__ == "__main__":
    main()