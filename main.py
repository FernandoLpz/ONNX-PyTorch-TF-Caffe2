from model import Train
from model import TorchModel
from src import Evaluation
from data import data_generator
		
if __name__ == '__main__':
	# Generate data samples for training and test
	x_train, x_test, y_train, y_test = data_generator(num_samples=100, visualize_plot=False)
	# Initialize training class
	torch_model = Train(x_train, y_train)
	# Train torch model
	torch_model.train_torch_model()
	# Save model
	torch_model.save_model()
	
	# Once the model is trained and saved as torch_model.onnx,
	# it will be loaded and evaluated with onnxruntime
	evaluation = Evaluation('onnx/torch_model.onnx')
	evaluation.torch_evaluation(x_test, y_test)
	evaluation.onnxruntime_evaluation(x_test, y_test)