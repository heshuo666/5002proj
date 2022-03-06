import numpy as np
import numpy.fft as nf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


# data preprocessing with fast fourier transform
def FFT(time_series,index):

    xf = range(len(time_series))

    # plot the figure of time series in time domain
    plt.subplot(221)
    plt.title('Time Domain', fontsize=10)
    plt.grid(linestyle=':')
    plt.plot(xf, time_series, c='orange', label='Raw')
    plt.legend()

    # transform the time series from time domain to frequency domain
    freqs = nf.fftfreq(len(time_series),300) #sample spacing
    fft_series = nf.fft(time_series)
    pows = np.abs(fft_series)

    # plot the figure of time series in frequency domain
    plt.subplot(222)
    plt.title('Frequency Domain', fontsize=10)
    plt.grid(linestyle=':')
    plt.plot(freqs, pows, c='mediumpurple', label='Raw')
    plt.legend()

    # low pass filter
    right_index = np.where(freqs >= 0.0005) # modify the threshold here
    left_index = np.where(freqs <= -0.0005)

    fft_series_copy = fft_series.copy()
    fft_series_copy[right_index] = 0
    fft_series_copy[left_index] = 0
    filter_pows = np.abs(fft_series_copy)

    # plot the figure of time series after filter in frequency domain
    plt.subplot(224)
    plt.grid(linestyle=':')
    plt.plot(freqs, filter_pows, c='steelblue', label='Filtered')
    plt.legend()

    # transform the filtered result back to time domain
    filter_time = nf.ifft(fft_series_copy).real

    # plot the figure of time series after filter in time domain
    plt.subplot(223)
    plt.grid(linestyle=':')
    plt.plot(xf, filter_time, c='forestgreen', label='Filtered')
    plt.legend()

    plt.figure()

    # calculate the residual
    res = time_series - filter_time

    filename1 = str(index) + "_FFT.txt"
    result_root1 = "../FFT_txt"
    np.savetxt(result_root1+"/"+filename1,res)

    plt.title('Residual', fontsize=10)
    plt.plot(xf, res, c='brown', label='Residual')
    plt.legend()

    result_root2 = "../FFT_figure"
    filename2 = str(index) + "_FFT.png"
    plt.savefig(result_root2+"/"+filename2)

    plt.show()


# load data
root_dir = "../data-sets/KDD-Cup/data"

path_list = os.listdir(root_dir)
path_list.sort()

for index, file in enumerate(tqdm(path_list)):
    if index >= 1 and index<=250:
        print(file)
        if file != ".DS_Store":
            time_series=np.loadtxt(root_dir+"/"+file) 
            FFT(time_series,index)