from torch import nn, tensor, utils, device, cuda, optim, long, save
from torch.autograd import Variable
import torch

class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()
    def forward(self, x):
        print(x.shape)
        return x


class RNNLayer(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(RNNLayer, self).__init__()
		self.hidden_dim = hidden_dim
		self.i2h = nn.Linear(input_dim + hidden_dim, hidden_dim)
		self.i2o = nn.Linear(input_dim + hidden_dim, output_dim)

	def forward(self, x, h):
		combined = torch.hstack((x,h))
		hn = self.i2h(combined)
		out = self.i2o(combined)

		return out, hn

	def init_hidden(self, batch_size):
		return torch.zeros(batch_size, self.hidden_dim)


class RNNModel(nn.Module):
	'''
	RNN module that can implement a LSTM
	Number of hiddend im and layer dim are parameters of the model
	'''
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
		super(RNNModel, self).__init__()

		# Number of hidden dimensions
		self.hidden_dim = hidden_dim

		# Number of hidden layers
		self.layer_dim = layer_dim

		# RNN
		#self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
		self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

		# Readout layer
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):

		# Initialize hidden state with zeros
		h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
		c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

		out, (hn, cn) = self.rnn(x, (h0, c0))
		out = self.fc(out[:, -1, :])

		return out

get = {
'MLP' : nn.Sequential(
    nn.Linear(5, 512),
    nn.ReLU(),
    nn.Linear(512, 1),
    ),
    
'Conv1D' : nn.Sequential(

    nn.Conv1d(1, kernel_size = 32, stride = 4, out_channels = 256),
    nn.LeakyReLU(),
    nn.MaxPool1d(kernel_size = 3, stride = 2),
    nn.Conv1d(256, kernel_size = 8, stride = 1, out_channels = 128),
    nn.LeakyReLU(),
    nn.MaxPool1d(kernel_size = 3, stride = 2),
    nn.Conv1d(128, kernel_size = 8, stride = 1, out_channels = 128),
    nn.LeakyReLU(),
    nn.MaxPool1d(kernel_size = 3, stride = 2),
    nn.Conv1d(128, kernel_size = 8, stride = 1, out_channels = 128),
    nn.LeakyReLU(),
    nn.MaxPool1d(kernel_size = 3, stride = 2),
    nn.Flatten(),
    nn.Linear(1024, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 2),
    nn.Softmax(-1)
    ),

'RNN' : nn.Sequential(

    nn.Linear(129, 512),
    nn.ReLU(),
    nn.Linear(512, 1),
    RNNModel(1, 1, 1, 1),
    )
}





