import numpy as np
import pandas as pd
import pathlib
import tqdm
rootpath = pathlib.Path('data-sets/KDD-Cup')
txt_dirpath = rootpath / 'data'  # Place the txt files in this directory

min_window_size = 20
growth_rate = 1.1
denom_threshold = 0.1
upper_threshold = 0.75
lower_threshold = 0.25
const_threshold = 0.05
min_coef = 0.5
small_quantile = 0.1
padding_length = 3
train_length = 10
use_gpu = True

names = [
    'orig_p2p',
    '1_order_p2p',
    '2_order_p2p',
    'orig_p2p_inv',
    '1_order_small',
    '1_order_large',
    '1_order_cross',
    '2_order_std',
    '2_order_std_inv',
]

# refer https://github.com/intellygenta
def compute_score(X, number, split, w):
    seq = pd.DataFrame(X, columns=['orig'])

    seq['1_order'] = seq['orig'].diff(1)
    seq['2_order'] = seq['1_order'].diff(1)

    for name in ['orig', '2_order']:
        seq[f'{name}_std'] = seq[name].rolling(w).std().shift(-w)

    for name in ['orig', '1_order', '2_order']:
        rolling_max = seq[name].rolling(w).max()
        rolling_min = seq[name].rolling(w).min()
        seq[f'{name}_p2p'] = (rolling_max - rolling_min).shift(-w)

    diff_abs = seq['1_order'].abs()
    cond = diff_abs <= diff_abs.quantile(small_quantile)
    seq['1_order_small'] = cond.rolling(w).mean().shift(-w)
    cond = diff_abs > diff_abs.quantile(small_quantile)
    seq['1_order_large'] = cond.rolling(w).mean().shift(-w)

    cond = seq['1_order'] * seq['1_order'].shift(1) < 0
    seq['1_order_cross'] = cond.rolling(w).mean().shift(-w)

    for name in ['orig_p2p', '2_order_std']:
        numer = seq[name].mean()
        denom = seq[name].clip(lower=numer * denom_threshold)
        seq[f'{name}_inv'] = numer / denom

    name = 'orig_p2p'
    mean = seq[name].mean()
    upper = mean * upper_threshold
    lower = mean * lower_threshold
    const = mean * const_threshold
    seq['coef'] = (seq[name] - lower) / (upper - lower)
    seq['coef'].clip(upper=1.0, lower=0.0, inplace=True)
    cond = (seq[name] <= const).rolling(2 * w).max().shift(-w) == 1
    seq.loc[cond, 'coef'] = 0.0
    if seq['coef'].mean() < min_coef:
        seq['coef'] = 0.0

    padding = w * padding_length
    seq['mask'] = 0.0
    seq.loc[seq.index[w:-w - padding], 'mask'] = 1.0
    seq['mask'] = seq['mask'].rolling(padding, min_periods=1).sum() / padding
    for name in names:
        seq[f'{name}_score'] = seq[name].rolling(w).mean() * seq['mask']

    return seq

lengths = np.loadtxt("data-sets/KDD-Cup/lengths.txt", dtype='int32')

results = []
for txt_filepath in sorted(txt_dirpath.iterdir()):

    X = np.loadtxt(txt_filepath)
    number = txt_filepath.stem.split('_')[0]
    split = int(txt_filepath.stem.split('_')[-1])
    print(f'\n{txt_filepath.name} {split}/{len(X)}', flush=True)

    # we re-write the code here, to let the range of the period larger to find a better result
    max_window_size = int(lengths[int(number)] * 0.05)
    size = int(np.log(max_window_size / min_window_size) / np.log(growth_rate)) + 1
    rates = np.full(size, growth_rate) ** np.arange(size)
    ws = (min_window_size * rates).astype(int)

    for w in tqdm.tqdm(ws):

        if w * train_length > split:
            continue

        seq = compute_score(X, number, split, w)

        for name in names:

            y = seq[f'{name}_score'].copy()

            cond = (y == y.rolling(w, center=True, min_periods=1).max())
            y.loc[~cond] = np.nan

            index1 = y.idxmax()
            value1 = y.max()

            if not np.isfinite(value1):
                continue

            if value1 <= 0.0:
                continue

            begin = index1 - w
            end = index1 + w
            if begin < split:
                continue

            y.iloc[begin:end] = np.nan
            index2 = y.idxmax()
            value2 = y.max()

            if value2 == 0:
                continue

            rate = value1 / value2
            results.append([number, w, name, rate, begin, end, index1, value1, index2, value2])

results = pd.DataFrame(results,
                       columns=['number', 'w', 'name', 'rate', 'begin', 'end', 'index1', 'value1', 'index2', 'value2'])

submission = results.loc[results.groupby('number')['rate'].idxmax(), ['w', 'name', 'rate', 'index1']]
submission.index = np.arange(len(submission)) + 1
submission.columns = ['window size', 'alg name', 'confidence', 'location']
submission.index.name = 'No.'
submission.to_csv('data-sets/KDD-Cup/result_stat_20_0.05_r1.1.csv')