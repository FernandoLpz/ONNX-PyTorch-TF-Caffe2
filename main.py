import onnx
import torch.onnx
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from model import TorchModel
from model import Train
from data import data_generator


def evaluation(model, x, y):
	with torch.no_grad():
		model.eval()
		ypred = model(x)
		return model.score(y, ypred)
		
if __name__ == '__main__':
	# Generate data samples for training and test
	x_train, x_test, y_train, y_test = data_generator(num_samples=100, visualize_plot=False)
	torch_model = Train(x_train, y_train)
	torch_model.save_model()
	
	######################################################################
	# print(f"Load into ONNNX")
	# # Load the ONNX model
	# model_onnx = onnx.load("onnx_model_name.onnx")
	# print(f"Print ONNX checker")
	# # Check that the IR is well formed
	# print(onnx.checker.check_model(model_onnx))
	# print(f"Print ONNX graph")
	# # Print a human readable representation of the graph
	# print(onnx.helper.printable_graph(model_onnx.graph))
	
	# import onnxruntime
	# ort_session = onnxruntime.InferenceSession("onnx_model_name.onnx")
	# print(f"This is the onnx session: {ort_session}")
	
	# # compute ONNX Runtime output prediction
	# ort_inputs = {ort_session.get_inputs()[0].name : dummy_input.detach().numpy()}
	# ort_outs = ort_session.run(None, ort_inputs)
	
	# print(f"Print onnx results: {ort_outs}")