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
import rrcf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

# refer to https://github.com/BTDLOZC-SJTU
class RRCF:
    def __init__(self, nums_trees, shingle_size, tree_size):
        self.num_trees = nums_trees
        self.shingle_size = shingle_size
        self.tree_size = tree_size
        self.forest = []
        self.p = 0
        self.index = 0

    def train(self, X, y=None):
        points = rrcf.shingle(X, size=self.shingle_size)
        points = np.vstack([point for point in points])
        n = points.shape[0]
        sample_size_range = (n // self.tree_size, self.tree_size)

        if n // self.tree_size == 0:
            print("Error! smaple size range is 0!")

        self.forest = []
        while len(self.forest) < self.num_trees:
            ixs = np.random.choice(n, size=sample_size_range,
                                   replace=False)
            trees = [rrcf.RCTree(points[ix], index_labels=ix) for ix in ixs]
            self.forest.extend(trees)

        avg_codisp = pd.Series(0.0, index=np.arange(n))
        index = np.zeros(n)

        for tree in self.forest:
            codisp = pd.Series({leaf: tree.codisp(leaf)
                                for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)

        avg_codisp /= index
        avg_codisp.index = X.iloc[(self.shingle_size - 1):].index
        avg_codisp = np.array([0] * (self.shingle_size - 1) + list(avg_codisp))

        p_0 = avg_codisp[np.where(y == 0)]
        p_1 = avg_codisp[np.where(y == 1)]
        if len(p_1) != 0:
            self.p = 0.6 * (p_0.mean() + p_1.mean())
        else:
            self.p = p_0.mean() + 9 * p_0.std()

    def predict(self, X):
        points = rrcf.shingle(X, size=self.shingle_size)
        points = np.vstack([point for point in points])
        n = points.shape[0]
        sample_size_range = (n // self.tree_size, self.tree_size)

        if n // self.tree_size == 0:
            print("Error! smaple size range is 0!")

        self.forest = []
        while len(self.forest) < self.num_trees:
            ixs = np.random.choice(n, size=sample_size_range,
                                   replace=False)
            trees = [rrcf.RCTree(points[ix], index_labels=ix) for ix in ixs]
            self.forest.extend(trees)

        avg_codisp = pd.Series(0.0, index=np.arange(n))
        index = np.zeros(n)

        for tree in self.forest:
            codisp = pd.Series({leaf: tree.codisp(leaf)
                                for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)

        avg_codisp /= index
        avg_codisp.index = X.iloc[(self.shingle_size - 1):].index
        avg_codisp = np.array([0] * (self.shingle_size - 1) + list(avg_codisp))

        return [0 if avg_codisp[index] <= self.p else 1 for index in range(len(avg_codisp))]

    def predict_proba(self, X):
        points = rrcf.shingle(X, size=self.shingle_size)
        points = np.vstack([point for point in points])
        n = points.shape[0]
        sample_size_range = (n // self.tree_size, self.tree_size)

        if n // self.tree_size == 0:
            print("Error! smaple size range is 0!")

        self.forest = []
        while len(self.forest) < self.num_trees:
            ixs = np.random.choice(n, size=sample_size_range,
                                   replace=False)
            trees = [rrcf.RCTree(points[ix], index_labels=ix) for ix in ixs]
            self.forest.extend(trees)

        avg_codisp = pd.Series(0.0, index=np.arange(n))
        index = np.zeros(n)

        for tree in self.forest:
            codisp = pd.Series({leaf: tree.codisp(leaf)
                                for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)

        avg_codisp /= index
        avg_codisp.index = X.iloc[(self.shingle_size - 1):].index
        avg_codisp = np.array([0] * (self.shingle_size - 1) + list(avg_codisp))

        return np.array(avg_codisp)

def _RRCF(tot_dfs, periods):
    for select in range(1, 251):
        window_size = periods[select]
        dfs_select = tot_dfs[select]

        sc = StandardScaler()
        # half period
        model = RRCF(100, int(window_size / 2), 4096)
        codisp = model.predict_proba(pd.Series(dfs_select))

        codisp = sc.fit_transform(codisp.reshape(-1, 1))

        np.savetxt(f"data-sets/KDD-Cup/RRCF/{select}_RRCF.txt", codisp, fmt='%.06f')

        plt.plot(codisp, color='g', label='anomaly score', linewidth=0.5)
        plt.title(f'{select} RRCF')
        plt.legend()
        plt.savefig(f'data-sets/KDD-Cup/RRCF/figures/{select}_RRCF_figure.png', dpi=1000, bbox_inches='tight')
        plt.close()

        print(f"{select} done")

def main():
    data_path = "data-sets/KDD-Cup/data"
    tot_dfs = dataLoad(data_path)

    periods = np.loadtxt("data-sets/KDD-Cup/period_list.txt", dtype='int32')
    _RRCF(tot_dfs, periods)

if __name__ == "__main__":
    main()