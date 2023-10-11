#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:06:18 2023

@author: cazaudo
"""
from pathlib import Path
import os
from OSmOSE.config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from OSmOSE.utils import make_path
from tqdm import tqdm
import sys
#from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Weather():

    def __init__(self, osmose_path_dataset, dataset, time_resolution_welch,sample_rate_welch,local=True):
        
        if isinstance(dataset,str):
            dataset = [dataset]
        
        self.path = Path(osmose_path_dataset)        
        self.dataset = dataset
        self.time_resolution_welch = time_resolution_welch
        self.sample_rate_welch = sample_rate_welch

    def wind_speed_estimation(self):

        for cur_dataset in self.dataset:        
            
            self.path = self.path.joinpath(cur_dataset)
            
            if not self.path.joinpath(OSMOSE_PATH.weather).exists():
                make_path(self.path.joinpath(OSMOSE_PATH.weather), mode=DPDEFAULT)        

            df=pd.read_csv(self.path.joinpath(OSMOSE_PATH.processed_auxiliary,str(self.time_resolution_welch)+'_'+str(self.sample_rate_welch),"aux_data.csv"),header=0)
    
            feature_matrix = pd.DataFrame({
                'SPL_filtered': df['SPL_filtered'],
                'InSituWIND': np.sqrt( df['interp_u10']**2 + df['interp_v10']**2)
            })
    
            feature_matrix = feature_matrix[pd.notnull(feature_matrix["InSituWIND"])]
            
            Y_wind= feature_matrix["InSituWIND"]
            X_wind=feature_matrix["SPL_filtered"]
            
            Y_categorical = pd.cut(Y_wind, [0,2.2,3.6,6,np.inf], right=False)
                
            x_train = X_wind.values
            y_train = Y_wind
    
            # min_max_scaler = MinMaxScaler()
            # x_train = min_max_scaler.fit_transform(x_train)    
                    
            my_dpi = 100
            fact_x = 0.7
            fact_y = 1
            fig, ax = plt.subplots(1, 1, figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
                                   dpi=my_dpi, constrained_layout=True)    
            #ax.scatter(x_train,y_train)
            data = pd.DataFrame(np.column_stack((x_train, y_train)), columns = 'x', 'y')
            sns.kdeplot(data = data, x = 'x', y = 'y', fill=True, ax = ax)

        
            z = np.polyfit(x_train, y_train, 2)
            fit = np.poly1d(z)
            ax.plot(np.sort(x_train), fit(np.sort(x_train)), label=fit, color="C3", alpha=1, lw=2.5  )  
            ax.legend([fit, ''])
            plt.xlabel('Relative SPL (dB)')
            plt.ylabel('ECMWF w10 (m/s)')
            plt.savefig(self.path.joinpath(OSMOSE_PATH.weather,"scatter_wind_model.png"), bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"Saving figure {self.path.joinpath(OSMOSE_PATH.weather,'scatter_wind_model.png')}")
                       
            with open( self.path.joinpath(OSMOSE_PATH.weather,'polynomial_law.txt'), 'w') as f:
                for item in z:
                    f.write("%s\n" % item)
                
            with open( self.path.joinpath(OSMOSE_PATH.weather,'min_max.txt'), 'w') as f:
                for item in [np.min(X_wind) , np.max(X_wind)]:
                    f.write("%s\n" % item)
                
    
            my_dpi = 100
            fact_x = 0.7
            fact_y = 1
            fig, ax = plt.subplots(1, 1, figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
                                   dpi=my_dpi, constrained_layout=True)        

            ax.scatter(y_train.values,fit(x_train))
            plt.xlim(np.min([np.min(y_train.values) , np.min(fit(x_train))])-1, np.max([np.max(y_train.values) , np.max(fit(x_train))])+1)
            plt.ylim(np.min([np.min(y_train.values) , np.min(fit(x_train))])-1, np.max([np.max(y_train.values) , np.max(fit(x_train))])+1)
            plt.savefig(self.path.joinpath(OSMOSE_PATH.weather,"scatter_ecmwf_model.png"), bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"Saving figure {self.path.joinpath(OSMOSE_PATH.weather,'scatter_ecmwf_model.png')}")
                  
            
                
    
            my_dpi = 100
            fact_x = 0.7
            fact_y = 1
            fig, ax = plt.subplots(1, 1, figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
                                   dpi=my_dpi, constrained_layout=True)              
            plt.plot(fit(x_train))
            plt.plot(y_train.values,'r')
            plt.autoscale(enable=True, axis='x', tight=True)
            ax.legend(['ecmwf', 'poly model'])
            plt.savefig(self.path.joinpath(OSMOSE_PATH.weather,"temporal_ecmwf_model.png"), bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"Saving figure {self.path.joinpath(OSMOSE_PATH.weather,'temporal_ecmwf_model.png')}")
                  
        
    def append_SPL_filtered(self,freq_min:int,freq_max:int):
        
        # get metadata from sepctrogram folder
        metadata_path = next(
            self.path.joinpath(OSMOSE_PATH.spectrogram,str(self.time_resolution_welch)+'_'+str(self.sample_rate_welch)).rglob("metadata.csv"), None
        )        
        metadata_spectrogram = pd.read_csv(metadata_path)
        
        df=pd.read_csv(self.path.joinpath(OSMOSE_PATH.processed_auxiliary,str(self.time_resolution_welch)+'_'+str(self.sample_rate_welch),"aux_data.csv"),header=0)
        
        SPL_filtered=[]
        for npz_path in tqdm(df['fn']):
            ltas = np.load(npz_path, allow_pickle = True)
        
            if freq_min != freq_max:
                pre_SPL = np.mean(ltas['Sxx'][0,np.argmin(abs(ltas['Freq']-freq_min)) : np.argmin(abs(ltas['Freq']-freq_max))])
            else:
                pre_SPL = np.mean(ltas['Sxx'][0,np.argmin(abs(ltas['Freq']-freq_min))])
                
            if metadata_spectrogram['spectro_normalization'][0] == "density":
                SPL_filtered.append(10 * np.log10((pre_SPL / (1e-12)) + (1e-20)))
            if metadata_spectrogram['spectro_normalization'][0] == "spectrum":
                SPL_filtered.append(10 * np.log10(pre_SPL + (1e-20)))
        
        df['SPL_filtered'] = SPL_filtered
        df.to_csv( self.path.joinpath(OSMOSE_PATH.processed_auxiliary,str(self.time_resolution_welch)+'_'+str(self.sample_rate_welch),"aux_data.csv"),
                index=False,
                na_rep="NaN"
            )
        
        
        
        
        
        
        
        
        
        

class benchmark_weather():

    def __init__(self, osmose_path_dataset, dataset,local=True):
        
        if not isinstance(dataset,list):
            print(f"Dataset should be multiple and defined within a list")
            sys.exit(0)
            
        self.path = Path(osmose_path_dataset)        
        self.dataset = dataset

        if not self.path.joinpath(OSMOSE_PATH.weather).exists():
            make_path(self.path.joinpath(OSMOSE_PATH.weather), mode=DPDEFAULT)   

    def compare_wind_speed_models(self):

        my_dpi = 80
        fact_x = 0.5
        fact_y = 0.9
        fig, ax = plt.subplots(1, 1, figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
                               dpi=my_dpi, constrained_layout=True)
        # datasetID=[]
        # for path in self.path.joinpath(OSMOSE_PATH.weather).iterdir():
        #     if path.is_dir():
        #         datasetID.append(path)

        veccol = ['r','b','g']
        
        print('Polynom coefs:')
        
        ct=0
        for dd in self.dataset:
            
            f = open(self.path.joinpath(dd,OSMOSE_PATH.weather,'polynomial_law.txt'), "r")
            xx=f.read()
            ll = [float(x) for x in xx.split('\n')[:-1]]
            
            p = np.poly1d(ll)
            
            print('-', dd , ' : ' , "{:.3f}".format(p[0]) , "/ {:.3f}".format(p[1]), "/ {:.3f}".format(p[2])  )
                
            x=np.arange(-20,0,0.1)
            y = p(x)
            plt.plot(x, y,c=veccol[ct])
            
            ct+=1
        
        plt.xlabel('Relative SPL (dB)')
        plt.ylabel('Estimated wind speed (m/s)')
        plt.legend(self.dataset)
        
        plt.savefig(self.path.joinpath(OSMOSE_PATH.weather,"compare_wind_model.png"), bbox_inches="tight", pad_inches=0)
        plt.close()






