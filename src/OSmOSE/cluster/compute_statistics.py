import os
import sys
import glob
import csv
import argparse

import soundfile as sf
import numpy as np
from scipy import signal

def Write_zscore_norma_params(*, input_dir: str, output_file: str, fmin_HighPassFilter: int, ind_min: int = 0, ind_max: int = -1):
    """Computes the normalization parameters for the Zscore normalisation of the dataset and writes it to a csv.
    
    This function can also be called from the command line. Type `get_zscore_params.py -h` to see the list of arguments.
    Will write the mean and standard deviation for each file in a .csv. To get the parameters, you need to read the csv and 
    compute the means of each columns.
    
    Parameters
    ----------
    input_dir: `str`
        The absolute path to the input directory. All audio files within this directory will be used.
    
    output_file: `str`
        The absolute path of the output csv file.
    
    fmin_HighPassFilter: `int`
        The minimum audio frequency under which data will be filtrated. If set to 0, then `sys.float_info.epsilon` will be used.
            
    ind_min: `int`
        The first file of the list to be processed. Default is 0.

    ind_max: `int`
        The last file of the list to be processed. Default is -1, meaning the entire list is processed.
        """
    all_files = sorted(glob.glob(os.path.join(input_dir , '*wav')))
    # If ind_max is -1, we go to the end of the list.
    wav_list = all_files[ind_min:ind_max if ind_max != -1 else len(all_files)]
    
    list_summaryStats = []

    for wav in wav_list:
        data, sample_rate = sf.read(wav)
        
        bpcoef=signal.butter(20, np.array([max(fmin_HighPassFilter,sys.float_info.epsilon), sample_rate/2-1]), fs=sample_rate, output='sos', btype='bandpass')
        data = signal.sosfilt(bpcoef, data)   
                
        list_summaryStats.append([wav, np.mean(data), np.std(data)])

    with open(output_file, 'w') as f:
        write = csv.writer(f)
        write.writerow(['filename','mean','std'])
        write.writerows(list_summaryStats)       


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python script to get the mean and standard deviation of audio files, used in zscore parameter calculation.")
    required = parser.add_argument_group('required arguments')
    required.add_argument("--input-dir", "-i", required=True, help="The input folder where the files to read are.")
    required.add_argument("--output-file", "-o", required=True, help="The csv file where the results will be written.")
    required.add_argument("--fmin-highpassfilter", "-fmin", required=True, type=int, help="The High Pass Filter.")
    parser.add_argument("--ind-min", "-min", type=int, default=0, help="The first file to consider. Default is 0.")
    parser.add_argument("--ind-max", "-max", type=int, default=-1, help="The last file to consider. -1 means consider all files from ind-min. Default is -1")

    args = parser.parse_args()

    Write_zscore_norma_params(input_dir=args.input_dir, output_file=args.output_file, fmin_HighPassFilter=args.fmin_highpassfilter, ind_min=args.ind_min, ind_max=args.ind_max)