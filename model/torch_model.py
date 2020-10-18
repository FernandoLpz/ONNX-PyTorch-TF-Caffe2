import torch
import torch.nn as nn

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