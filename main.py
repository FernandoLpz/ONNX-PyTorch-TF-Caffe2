from model import Train
from model import TorchModel
from src import Evaluation
from data import data_generator
		
if __name__ == '__main__':
	# Generate data samples for training and test
	x_train, x_test, y_train, y_test = data_generator(num_samples=100, visualize_plot=False)
	torch_model = Train(x_train, y_train)
	torch_model.train_torch_model()
	torch_model.save_model()
	
	evaluation = Evaluation('onnx/torch_model.onnx')
	evaluation.torch_evaluation(x_test, y_test)
	evaluation.onnx_runtime_evaluation(x_test, y_test)