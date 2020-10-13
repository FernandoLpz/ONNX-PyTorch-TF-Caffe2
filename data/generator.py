import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def data_generator(num_samples=1000, visualize_plot=False):

	# Defines x-axis
	x1_axis = np.random.rand(num_samples)
	x2_axis = np.random.rand(num_samples)
	
	# Defines y-axis under gauss distribution
	mu1, sigma1 = 0.3, 0.01
	y1_axis = np.random.normal(mu1, sigma1, num_samples)
	mu2, sigma2 = 0.7, 0.01
	y2_axis = np.random.normal(mu2, sigma2, num_samples)
	
	# Plot generated data
	if visualize_plot:
		plt.plot(x1_axis, y1_axis, 'o', color='blue')
		plt.plot(x2_axis, y2_axis, 'o', color='green')
		plt.show()
	
	# Stacks class = 0
	x1 = np.vstack((x1_axis, y1_axis))
	x1 = x1.reshape((x1.shape[1], x1.shape[0]))
	y1 = np.zeros(x1.shape[0])
	
	# Stacks class = 1
	x2 = np.vstack((x2_axis, y2_axis))
	x2 = x2.reshape((x2.shape[1], x2.shape[0]))
	y2 = np.ones(x2.shape[0])
	
	# Stacks both classes 
	x = np.vstack((x1, x2))
	y = np.hstack((y1, y2))
	
	# y = y.reshape((y.shape[0], 1))
	
	# Split data
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=12)
	
	return x_train, x_test, y_train, y_test