import torch
import torch.onnx
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from data import data_generator

	
class TorchModel(nn.ModuleList):
	def __init__(self):
		super(TorchModel, self).__init__()
		self.linear_1 = nn.Linear(2, 12)
		self.linear_2 = nn.Linear(12, 1)
		
	def forward(self, x):
		out = self.linear_1(x)  
		out = torch.tanh(out)
		out = self.linear_2(out)
		out = torch.sigmoid(out)
		
		return out
		
	@staticmethod
	def score(y_true, y_pred):
		tp, fp = 0, 0
		for ytrue, ypred in zip(y_true, y_pred):
			if (ytrue == 1) and (ypred >= 0.5):
				tp += 1
			elif (ytrue == 0) and (ypred < 0.5):
				fp += 1
		
		return (tp+fp)/len(y_true)

def training(x ,y):
	model = TorchModel()
	optimizer = optim.Adam(model.parameters(), lr=0.1)
	criterion = nn.BCELoss()
	
	batch_size = 2
	num_batches = int(len(x) / batch_size)
	
	for epoch in range(100):
		model.train()
		for batch in range(num_batches):
			try:
				x_batch = x[(batch * batch_size) : ((batch+1) * batch_size)]
				y_batch = y[(batch * batch_size) : ((batch+1) * batch_size)]
			except:
				x_batch = x[(batch * batch_size) :]
				y_batch = y[(batch * batch_size) :]
				
			x_batch = torch.from_numpy(x_batch).type(torch.FloatTensor)
			y_batch = torch.from_numpy(y_batch).type(torch.FloatTensor)
			out = model(x_batch)
			loss = criterion(out.view(-1), y_batch)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
		if epoch % 10 == 0:
			with torch.no_grad():
				model.eval()
				xtrain = torch.from_numpy(x).type(torch.FloatTensor)
				ypred = model(xtrain)
				score = model.score(y, ypred.detach().numpy()) 
			print("Epoch: %d,  loss: %.5f, score: %.5f " % (epoch, loss.item(), score))
	
	print(f"Exporting model")
	dummy_input = Variable(torch.randn(2, 12)) 
	torch.onnx.export(model, dummy_input, "onnx_model_name.onnx")
			
	return model

def evaluation(model, x, y):
	with torch.no_grad():
		model.eval()
		x = torch.from_numpy(x).type(torch.FloatTensor)
		ypred = model(x)
		return model.score(y, ypred.detach().numpy())
		
if __name__ == '__main__':
	x_train, x_test, y_train, y_test = data_generator(num_samples=100, visualize_plot=True)
	model = training(x_train, y_train)
	train_score = evaluation(model, x_train, y_train)
	test_score = evaluation(model, x_test, y_test)
	print(f"\nTrain score: {train_score}\nTest score: {test_score}")