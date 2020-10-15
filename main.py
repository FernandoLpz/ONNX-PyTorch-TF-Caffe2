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
	
	######################################################################
	# print(f"Save pytorch model ")
	# torch.save(torch_model.state_dict(),"model.pt")
	# print(f"Model dicth:\n{torch_model.state_dict()}")
	# print(f"Init again torch model")
	# trained_model = TorchModel()
	# print(f"Load weights")
	# trained_model.load_state_dict(torch.load('model.pt'))
	dummy_input = Variable(torch.randn(1,2))
	# print(f"Export into onnx")
	input_names = ["Input Layer", "Layer 1 (weights)", "Layer 1 (biases)", "Layer 2 (weights)", "Layer 2 (biases)" ]
	output_names = [ "Output Layer" ]
	torch.onnx.export(torch_model, dummy_input, "onnx_model_name.onnx", verbose=True, opset_version=10, input_names=input_names, output_names=output_names)
	
	
	######################################################################
	print(f"Load into ONNNX")
	# Load the ONNX model
	model_onnx = onnx.load("onnx_model_name.onnx")
	print(f"Print ONNX checker")
	# Check that the IR is well formed
	print(onnx.checker.check_model(model_onnx))
	print(f"Print ONNX graph")
	# Print a human readable representation of the graph
	print(onnx.helper.printable_graph(model_onnx.graph))
	
	import onnxruntime
	ort_session = onnxruntime.InferenceSession("onnx_model_name.onnx")
	print(f"This is the onnx session: {ort_session}")
	
	# compute ONNX Runtime output prediction
	ort_inputs = {ort_session.get_inputs()[0].name : dummy_input.detach().numpy()}
	ort_outs = ort_session.run(None, ort_inputs)
	
	print(f"Print onnx results: {ort_outs}")