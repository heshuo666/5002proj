import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')


def dataLoad(data_path):
    dir_list = os.listdir(data_path)
    partitions = [0]
    lengths = [0]

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

        partition = int(filename.split('_')[-1].split('.')[0])
        partitions.append(partition)
        lengths.append(len(df))
        index += 1

    return partitions, lengths

def main():
    data_path = "data-sets/KDD-Cup/data"
    partitions, lengths = dataLoad(data_path)

    partitions = np.array(partitions)
    lengths = np.array(lengths)

    np.savetxt("data-sets/KDD-Cup/partitions.txt", partitions[0:], fmt='%d')
    np.savetxt("data-sets/KDD-Cup/lengths.txt", lengths[0:], fmt='%d')

if __name__ == "__main__":
    main()