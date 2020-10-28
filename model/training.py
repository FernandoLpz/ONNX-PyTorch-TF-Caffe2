import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from src import Evaluation
from model.torch_model import TorchModel

class Train:

	def __init__(self, x, y):
		self.model = TorchModel()
		self.x = torch.from_numpy(x).type(torch.FloatTensor)
		self.y = torch.from_numpy(y).type(torch.FloatTensor)
		
		# Fixed parameters
		self.batch_size = 2
		self.num_epochs = 100
		self.learning_rate = 0.1
		self.onnx_path = f"onnx/"
		
	def train_torch_model(self):
	
		# Init optimizer and loss function
		optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
		criterion = nn.BCELoss()
		
		num_batches = int(len(self.x) / self.batch_size)
		
		# Start training
		for epoch in range(self.num_epochs):
		  # Set model in training mode
			self.model.train()
			for batch in range(num_batches):
				try:
					x_batch = self.x[(batch * self.batch_size) : ((batch+1) * self.batch_size)]
					y_batch = self.y[(batch * self.batch_size) : ((batch+1) * self.batch_size)]
				except:
					x_batch = self.x[(batch * self.batch_size) :]
					y_batch = self.y[(batch * self.batch_size) :]
				
				# Feed model
				out = self.model(x_batch)
				# Calculate loss
				loss = criterion(out.view(-1), y_batch)
				# Clean gradients
				optimizer.zero_grad()
				# Backpropagate loss
				loss.backward()
				# Update weights
				optimizer.step()
				
			# Shows metrics every 10 epochs
			if epoch % 10 == 0:
				with torch.no_grad():
					# Turn model into evaluation mode
					self.model.eval()
					ypred = self.model(self.x)
					score = Evaluation.score(self.y, ypred) 
				print("Epoch: %d,  Loss: %.5f, Accuracy: %.5f " % (epoch, loss.item(), score))
				
		pass
		
	def save_model(self, dummy_input=False, verbose=False):
		# If true, a generic input is generated
		# otherwise the training tensor is used
		if dummy_input:
			x = Variable(torch.randn(self.batch_size, 2))
		else:
			x = self.x
		
		input_names = ["Input","Linear 1 (weights)","Linear 1 (biases)","Linear 2 (weights)","Linear 2 (biases)" ]
		output_names = [ "Output Layer" ]
	
		# Save the model as onnx
		torch.onnx.export(self.model, x, 
								f"{self.onnx_path}torch_model.onnx", 
								verbose=verbose, 
								input_names=input_names, 
								output_names=output_names,
								keep_initializers_as_inputs=True,
								# dynamic_axis allow us to define which dimentsion size
								# will vary when loading the model with other framework 
								# (e.g. onnnxruntime, keras, caffe2, etc.)
								dynamic_axes={"Input": {0: "batch"}})
