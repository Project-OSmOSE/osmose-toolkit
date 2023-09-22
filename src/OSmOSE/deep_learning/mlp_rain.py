import numpy as np
import os
import pandas as pd
import torch
import soundfile as sf
import utils_dl as u
import random
from tqdm import tqdm
from torch import utils
import torch.nn as nn
import models as m
import visdom




################# TRAINING AND TESTING #####################

def train(model, dataloader, test_loader, criterion, optimizer, num_epochs, accuracy, plotter = None):

	for epoch in range(num_epochs):
		acc_batch = []
		for batch in tqdm(dataloader):

			optimizer.zero_grad()

			data, labels = batch
			outputs = model(data)
			loss = criterion(outputs.squeeze(), labels)

			loss.backward()
			optimizer.step()

			outputs, labels = outputs.squeeze().detach().numpy(), labels.numpy()
			acc_batch.append(np.sum(abs(outputs-labels) < 0.1)/len(labels))
		accuracy.append(np.mean(acc_batch))
		
		if type(plotter) != type(None) :
			plotter.plot('accuracy', 'train', 'accuracy', epoch, accuracy[-1])
			plotter.plot('loss', 'train', 'loss', epoch, loss.detach().numpy())
		if (epoch+1) % 1 == 0:
			test(model, test_loader, epoch, plotter)
			model.train()
			print(f'Epoch {epoch+1}/{num_epochs}, Loss = {loss.item():.4f}, Accuracy = {accuracy[-1]:.4f}')



def train_rnn(model, rnn, dataloader, test_loader, criterion, optimizer, num_epochs, accuracy, plotter = None):

	for epoch in range(num_epochs):
		acc_batch = []
		for batch in dataloader:

			optimizer.zero_grad()

			data, labels = batch
			print(data.device)
			#For pytorch LSTM
			outputs = model(data)
			
			'''
			#For custom build RNN
			hn = rnn.init_hidden(len(data))
			for i in range(data.size(1)):     #Look at every element of the sequence one by one and feed it to RNN
				outputs, hn = rnn(outputs, hn)'''
				
			loss = criterion(outputs.squeeze(), labels)
			loss.backward()
			optimizer.step()
			outputs, labels = outputs.squeeze().detach().numpy(), labels.numpy()
			acc_batch.append(np.sum(abs(outputs-labels) < 0.1)/len(labels))
			
		accuracy.append(np.mean(acc_batch))
		if type(plotter) != type(None):
			plotter.plot('accuracy', 'train', 'accuracy', epoch, accuracy[-1])
			plotter.plot('loss', 'train', 'loss', epoch, loss.detach().numpy())
		if (epoch+1) % 3 == 0:
			test_rnn(model, rnn, test_loader, epoch)
			model.train()
			print(f'Epoch {epoch+1}/{num_epochs}, Loss = {loss.item():.4f}, Accuracy = {accuracy[-1]:.4f}')



def test(model, dataloader, epoch, plotter):
	model.eval()
	acc_test, loss_test = [], []
	all_preds, all_labels = [], []
	with torch.no_grad():
		for batch in dataloader:
			data, labels = batch
			outputs = model(data)
			loss_test.append(criterion(outputs.squeeze(), labels))
			outputs, labels = outputs.squeeze().detach().numpy(), labels.numpy()
			all_preds.extend(outputs)
			all_labels.extend(labels)
			acc_test.append(np.sum(abs(outputs-labels) < 1)/len(labels))	
		np.save('results/'+str(epoch+1), np.vstack((all_preds, all_labels)))

		plotter.plot('accuracy', 'test', 'accuracy', epoch, acc_test[-1])
		plotter.plot('loss', 'test', 'loss', epoch, loss_test[-1])
	
def test_rnn(model, rnn, dataloader, epoch):
	model.eval()
	acc_test = []
	all_preds, all_labels = [], []
	with torch.no_grad():
		for batch in dataloader:
			data, labels = batch
			
			outputs = model(data)
			
			'''#hn = rnn.init_hidden(len(data))
			for i in range(data.size(1)):
				#outputs, hn = rnn(outputs, hn)'''

			outputs, labels = outputs.squeeze().detach().numpy(), labels.numpy()
			all_preds.extend(outputs)
			all_labels.extend(labels)
			acc_test.append(np.sum(abs(outputs-labels) < 1)/len(labels))	
		np.save('results/'+str(epoch+1), np.vstack((all_preds, all_labels)))




##########################################################################

'''
# set the device (CPU or GPU)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


# create the model and move it to the specified device
model = m.RNNModel(5, 512, 1, 1)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f'Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} params')
#model = m.get['MLP']
model.to(device)
rnn = m.RNNLayer(1,1,1).to(device)


# create the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.000)

ses = pd.read_csv('/run/media/grosmaan/LaCie/individus_brut/individus/ses', header = None).to_numpy().reshape(12)

trainset = pd.DataFrame()
for elem in ses[[0,1,3,5,7,8,9]]:
	aux_df = pd.read_csv(elem+'/aux_data.csv')[['tp', 'depth']] #[['lon', 'lat', 'bathy', 'depth', 'shore_dist', 'era']]
	spl_df = pd.read_csv(elem+'/spl_data_cal.csv')
	temp = pd.concat((aux_df, spl_df), axis = 1)
	trainset = pd.concat((trainset, temp))
trainset = trainset.dropna(subset = ['tp', '5000', '8000', '10000', '12500', '15000'])
trainset = trainset[trainset.depth > 70]
#trainset = trainset[trainset['15000'] < trainset['15000'].quantile(0.99)]
trainset = trainset[::5]    # USE LSTM WITH MORE VARIATIONS IN WIND SPEED, wind speed doesnt change much over 5 mn
trainset = trainset.reset_index()
trainset.tp = trainset.tp * 1000


testset = pd.DataFrame()
for elem in ses[[10,11]]:
	aux_df = pd.read_csv(elem+'/aux_data.csv')[['tp', 'depth']] #[['lon', 'lat', 'bathy', 'depth', 'shore_dist', 'era']]
	spl_df = pd.read_csv(elem+'/spl_data_cal.csv')
	temp = pd.concat((aux_df, spl_df), axis = 1)
	testset = pd.concat((testset, temp))
testset = testset.dropna(subset = ['tp', '5000', '8000', '10000', '12500', '15000'])
testset = testset[testset.depth > 70]
#testset = testset[testset['15000'] < testset['15000'].quantile(0.99)]
testset = testset[::5]
testset = testset.reset_index()
testset.tp = testset.tp*1000

global plotter
from visdom import Visdom
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
plotter = VisdomLinePlotter(env_name='MLP_wind')


print(model)
train_loader = utils.data.DataLoader(u.SES(trainset, 'tp', seq_length=5), batch_size = 64, shuffle = True)
test_loader = utils.data.DataLoader(u.SES(testset, 'tp', seq_length=5), batch_size = 64, shuffle=False)
# train the model
train_rnn(model, rnn, train_loader, test_loader, criterion, optimizer, num_epochs=1000, accuracy = []) #, plotter=plotter)
'''




