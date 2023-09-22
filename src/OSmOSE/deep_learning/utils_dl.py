import numpy as np
import pandas as pd
import scipy.signal as sg
from torch import nn, tensor, utils, device, cuda, optim, long, save
from torch.utils import data
from torchvision import ops
from tqdm import tqdm
import soundfile as sf
import random
from matplotlib import pyplot as plt
import calendar, time, datetime, torch
import scipy.interpolate as si
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import os


def data_augmentation(welch):
	welch = welch.reshape(-1, welch.shape[-1])
	wnoise = 0.002
	wmax = 1.1*np.max(abs(welch))
	nwelch, wlen = welch.shape
	warping = True
	da = True
	if da :
		amp_offset, amp_sin =  np.random.uniform(-0.02*wmax, 0.02*wmax, (2, nwelch))
		freq_sin, phase = np.random.uniform(1e-8, 3/10, nwelch), np.random.uniform(0, np.pi, nwelch)
		welch += np.vstack([np.random.normal(amp_offset[i], wmax*wnoise, wlen) for i in range(nwelch)])
		welch += np.vstack([amp_sin[i]*np.sin(2*np.pi*freq_sin[i]*np.linspace(0,5, wlen)+phase[i]) for i in range(nwelch)])
	if warping and wlen > 50:
		for i in range(nwelch):
			warp = np.random.randint(3, wlen/10, nwelch)
			pos_squeeze = np.random.randint(0, wlen-warp, nwelch)
			warp_ind = np.unique(np.random.randint(pos_squeeze, pos_squeeze+warp, warp//2))
			pos1, pos2 = np.random.randint(0, wlen-len(warp_ind), 2)
			if pos1 == pos2:
				pos1+=1
			temp_welch = np.delete(welch[i], warp_ind)
			welch[i] = np.concatenate((temp_welch[:min(pos1, pos2)], sg.resample(temp_welch[min(pos1, pos2) : max(pos1, pos2)], abs(pos2-pos1)+len(warp_ind)), temp_welch[max(pos1, pos2):]))
	return welch




########################################## DATALOADER FOR PYTORCH ############################################################

##############################################################################################################################



class Welch_RNN(data.Dataset):

	def __init__(self, df, variable, seq_length, path, da = False):
		super(data.Dataset, self)
		self.df = df
		self.path = path
		self.variable = variable
		self.seq_length = seq_length
		if da:
			self.f = data_augmentation
		else:
			self.f = lambda x: x

	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, idx):
		welch = np.load(os.path.join(self.path, self.df.fn.iloc[idx]))['Sxx']
		for i in range(1, self.seq_length):
			welch = np.vstack((welch, np.load(os.path.join(self.path, self.df.fn.iloc[idx-i]))['Sxx']))
		return torch.FloatTensor(welch), torch.tensor(self.df[self.variable][idx], dtype=torch.float)


class LoadWelch(data.Dataset):

	def __init__(self, df, path, da = False):
		super(data.Dataset, self)
		self.df = df
		self.path = path
		if da:
			self.f = data_augmentation
		else:
			self.f = lambda x: x

	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, idx):
		welch = self.f(np.log(np.load(self.path+self.df['0'][idx][:-4]+'_5mn.npy')[1]))
		return torch.FloatTensor(welch), torch.tensor(self.df['era5'][idx], dtype=torch.float)

class SES(data.Dataset):
	
	def __init__(self, df, variable, seq_length):
		self.df = df
		self.seq_length = seq_length
		self.variable = variable 

	def __len__(self):
		return self.df.shape[0]
	
	def __getitem__(self, idx):
		spl = torch.FloatTensor(self.df[['8000', '10000', '12500', '15000', 'depth']].loc[idx-self.seq_length+1:idx].to_numpy())
		if len(spl) == self.seq_length:
			return spl, torch.tensor(self.df[self.variable].loc[idx], dtype=torch.float)
		else:
			spl = torch.cat((torch.zeros(self.seq_length-len(spl), spl.size(1)), spl))
			return spl, torch.tensor(self.df[self.variable].loc[idx], dtype=torch.float)

		#aux = torch.FloatTensor(self.df[['depth', 'lat', 'lon', 'bathy', 'shore_dist']].loc[idx])
		
		#return torch.cat((spl, aux)), torch.tensor(self.df['era'].loc[idx], dtype=torch.float)


class LoadWav(data.Dataset):
    
	size = 10
	
	def __init__(self, df, path, timestamps, da = False):
		super(data.Dataset, self)
		self.df = df 
		self.path = path
		self.timestamps = timestamps
		if da:
			self.f = data_augmentation
		else:
			self.f = lambda x: x

	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, idx):
		#fs = sf.info(self.path + self.df[idx, 1]).samplerate
		data = self.df.iloc[idx]
		start_time = self.timestamps[:,1][self.timestamps[:,0] == data['fn']].item()
		fs = sf.info(self.path[idx]+data['fn']).samplerate
		sig, fs = sf.read(self.path[idx]+data['fn'], start = int((data['time']-start_time)*fs), stop = int((data['time']-start_time+self.size)*fs))
		welch = self.f(np.log(sg.welch(sig, fs, 'hann', 2048, 1000)))
		return torch.FloatTensor(welch[1]), torch.tensor(data['era_cl'], dtype=torch.float)


'''
def collate_fn(batch):
	#collate_fn is used to do operations on batch in torch's Dataloader
	# Sort the batch in descending order of sequence length
	batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
	   
	max_length = len(batch[0][0])
	# Initialize tensors to hold the padded sequences

	# Fill in the sequences
	X, label = [], []
	for i, (sequence, era) in enumerate(batch):
		if len(sequence) == max_length:
			X.append(torch.tensor(sequence))
			label.append(era)
		else:
			temp_sequence = [0]*(max_length - len(sequence))
			X.append(torch.tensor([temp_sequence.extend(sequence)]))
			label.append(era)

	X = torch.tensor(X)
	label = torch.tensor(label)

	return (X, label)
'''
