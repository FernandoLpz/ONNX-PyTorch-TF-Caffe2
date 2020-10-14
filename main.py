import onnx
import torch.onnx
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from models import TorchModel
from data import data_generator

def train_torch_model(x ,y):

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

			out = model(x_batch)
			loss = criterion(out.view(-1), y_batch)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
		if epoch % 10 == 0:
			with torch.no_grad():
				model.eval()
				ypred = model(x)
				score = model.score(y, ypred) 
			print("Epoch: %d,  loss: %.5f, accuracy: %.5f " % (epoch, loss.item(), score))
			
	return model

def evaluation(model, x, y):
	with torch.no_grad():
		model.eval()
		ypred = model(x)
		return model.score(y, ypred)
		
if __name__ == '__main__':
	# Generate data samples for training and test
	x_train, x_test, y_train, y_test = data_generator(num_samples=100, visualize_plot=False)
	
	# Transform numpy arrays into torch tensors
	x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
	y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
	
	# Train torch model
	torch_model = train_torch_model(x_train, y_train)
	# Evaluates torch model
	train_score = evaluation(torch_model, x_train, y_train)
	test_score = evaluation(torch_model, x_test, y_test)
	print(f"Train acc: {train_score}, Test acc: {test_score}")
	
	print(f"Exporting model")
	torch.onnx.export(torch_model, x_train, "onnx_model_name.onnx")
	
	print(f"Loading model")
	model_onnx = onnx.load("onnx_model_name.onnx")
	
	# Print a human readable representation of the graph
	print(onnx.helper.printable_graph(model_onnx.graph))
	
	import caffe2.python.onnx.backend as backend
	from caffe2.python import core, workspace
	# Generate data samples for training and test
	x_train, x_test, y_train, y_test = data_generator(num_samples=100, visualize_plot=False)
	inferred_model = onnx.shape_inference.infer_shapes(model_onnx)
	print(inferred_model)
	# rep = backend.prepare(model_onnx, device="CPU")
	# output = rep.run(x_train.astype(np.float32))
	# print(output)