import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import sys

def ensemble(stat_confidence, stat_location, partitions):
    final = [0]
    for select in range(1, 251):
        lengths = []

        FFT_select = np.loadtxt(f'data-sets\KDD-Cup\FFT\{select}_FFT.txt', dtype=float)
        lengths.append(FFT_select.shape[0])

        MP_select = np.loadtxt(f'data-sets\KDD-Cup\MP\{select}_MP0.txt', dtype=float)
        lengths.append(MP_select.shape[0])

        RRCF_select = np.loadtxt(f'data-sets\KDD-Cup\RRCF\{select}_RRCF.txt', dtype=float)
        lengths.append(RRCF_select.shape[0])

        SW_select = np.loadtxt(f'data-sets\KDD-Cup\sliding_window\{select}_SW_diff.txt', dtype=float)
        lengths.append(SW_select.shape[0])

        SR_select = np.loadtxt(f'data-sets\KDD-Cup\SR\{select}_SR.txt', dtype=float)
        lengths.append(SR_select.shape[0])

        VMD_select = np.loadtxt(f'data-sets\KDD-Cup\VMD\DC=0_file\\No.{select}_VMD.txt', dtype=float)
        lengths.append(VMD_select.shape[0])

        length_select = min(lengths)

        partition_select = partitions[select]

        FFT_select = np.absolute(FFT_select[partition_select: length_select])
        MP_select = np.absolute(MP_select[partition_select: length_select])
        RRCF_select = np.absolute(RRCF_select[partition_select: length_select])
        SW_select = np.absolute(SW_select[partition_select: length_select])
        SR_select = np.absolute(SR_select[partition_select: length_select])
        VMD_select = np.absolute(VMD_select[partition_select: length_select])

        test_length_real = length_select - partition_select

        window_size = int(test_length_real / 30)

        FFT_cands = []
        MP_cands = []
        RRCF_cands = []
        SW_cands = []
        SR_cands = []
        VMD_cands = []

        for index in range(0, test_length_real, window_size):
            FFT_cands.append(np.max(FFT_select[index: index + window_size]))
            MP_cands.append(np.max(MP_select[index: index + window_size]))
            RRCF_cands.append(np.max(RRCF_select[index: index + window_size]))
            SW_cands.append(np.max(SW_select[index: index + window_size]))
            SR_cands.append(np.max(SR_select[index: index + window_size]))
            VMD_cands.append(np.max(VMD_select[index: index + window_size]))

        FFT_cands = np.array(FFT_cands)
        MP_cands = np.array(MP_cands)
        RRCF_cands = np.array(RRCF_cands)
        SW_cands = np.array(SW_cands)
        SR_cands = np.array(SR_cands)
        VMD_cands = np.array(VMD_cands)

        FFT_confidence = np.sort(FFT_cands)[-1] / np.sort(FFT_cands)[-2]
        MP_confidence = np.sort(MP_cands)[-1] / np.sort(MP_cands)[-2]
        RRCF_confidence = np.sort(RRCF_cands)[-1] / np.sort(RRCF_cands)[-2]
        SW_confidence = np.sort(SW_cands)[-1] / np.sort(SW_cands)[-2]
        SR_confidence = np.sort(SR_cands)[-1] / np.sort(SR_cands)[-2]
        VMD_confidence = np.sort(VMD_cands)[-1] / np.sort(VMD_cands)[-2]


        stat_confidence_select = stat_confidence[select - 1]


        max_confidence = max([FFT_confidence, MP_confidence, RRCF_confidence, SW_confidence, SR_confidence, VMD_confidence])


        if max_confidence < stat_confidence_select:
            print(f"{select}", stat_location[select - 1], 0)
            final.append(stat_location[select - 1])
        else:
            FFT_stand = (FFT_select - np.mean(FFT_select)) / np.std(FFT_select)
            FFT_norm = (FFT_stand - np.min(FFT_stand)) / (np.max(FFT_stand) - np.min(FFT_stand))

            MP_stand = (MP_select - np.mean(MP_select)) / np.std(MP_select)
            MP_norm = (MP_stand - np.min(MP_stand)) / (np.max(MP_stand) - np.min(MP_stand))

            RRCF_stand = (RRCF_select - np.mean(RRCF_select)) / np.std(RRCF_select)
            RRCF_norm = (RRCF_stand - np.min(RRCF_stand)) / (np.max(RRCF_stand) - np.min(RRCF_stand))

            SW_stand = (SW_select - np.mean(SW_select)) / np.std(SW_select)
            SW_norm = (SW_stand - np.min(SW_stand)) / (np.max(SW_stand) - np.min(SW_stand))

            SR_stand = (SR_select - np.mean(SR_select)) / np.std(SR_select)
            SR_norm = (SR_stand - np.min(SR_stand)) / (np.max(SR_stand) - np.min(SR_stand))

            VMD_stand = (VMD_select - np.mean(VMD_select)) / np.std(VMD_select)
            VMD_norm = (VMD_stand - np.min(VMD_stand)) / (np.max(VMD_stand) - np.min(VMD_stand))

            total = FFT_confidence ** 2 + MP_confidence ** 2 + RRCF_confidence ** 2 + SW_confidence ** 2 + SR_confidence ** 2 + VMD_confidence ** 2

            FFT_confidence2 = FFT_confidence ** 2 / total
            MP_confidence2 = MP_confidence ** 2 / total
            RRCF_confidence2 = RRCF_confidence ** 2 / total
            SW_confidence2 = SW_confidence ** 2 / total
            SR_confidence2 = SR_confidence ** 2 / total
            VMD_confidence2 = VMD_confidence ** 2 / total

            FINAL = FFT_norm * FFT_confidence2 + MP_norm * MP_confidence2 + RRCF_norm * RRCF_confidence2 + SW_norm * SW_confidence2 + SR_norm * SR_confidence2 + VMD_norm * VMD_confidence2

            print(f"{select}", np.argmax(FINAL) + partition_select + 1, 1)
            final.append(np.argmax(FINAL) + partition_select + 1)

    final = np.array(final)
    np.savetxt(f"data-sets/KDD-Cup/my_final_30.txt", final, fmt='%d')


def main():
    stat_data = pd.read_csv('data-sets\KDD-Cup\\result_stat.csv')

    stat_confidence = stat_data['confidence']

    stat_location = stat_data['location']

    partitions = np.loadtxt("data-sets/KDD-Cup/partitions.txt", dtype='int32')

    ensemble(stat_confidence, stat_location, partitions)

if __name__ == "__main__":
    main()