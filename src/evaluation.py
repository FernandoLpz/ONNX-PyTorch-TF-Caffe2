import onnx
import onnxruntime
import caffe2.python.onnx.backend as onnx_caffe2_backend
from onnx_tf.backend import prepare
from mxnet.contrib import onnx as onnx_mxnet


class Evaluation:
   def __init__(self, x, y, onnx_model, print_graph=True):
      # In order to evaluate the trained model, it will be required
      # to fix the datatypes. As default, the numpy array saved the test set
      # as float64, however it is required to feed the model with a float32 datatype
      self.x = x.astype('float32')
      self.y = y
      self.onnx_model = onnx_model
      if print_graph:
         # The onnx model is loaded and it is printed
         # in a human-readble format
      	model_onnx = onnx.load(self.onnx_model)
      	print(onnx.helper.printable_graph(model_onnx.graph))
      	
   @staticmethod
   def score(y_true, y_pred):
      tp, fp = 0, 0
      for ytrue, ypred in zip(y_true, y_pred):
         if (ytrue == 1) and (ypred >= 0.5):
            tp += 1
         elif (ytrue == 0) and (ypred < 0.5):
            fp += 1
      return (tp  + fp)/len(y_true)
   
   def onnxruntime_evaluation(self):  
      # Lauch the session
      onnx_session = onnxruntime.InferenceSession(self.onnx_model)
      # Assigns the input numpy array as input vector for the onnx session
      onnx_inputs = {onnx_session.get_inputs()[0].name : self.x}
      # Inference
      onnx_predictions = onnx_session.run(None, onnx_inputs)
      
      # Calculate accuracy
      score = Evaluation.score(y_true=self.y, y_pred=onnx_predictions[0])
      print(f"ONNX Runtime accuracy: {score}")
      
      pass
      
   def caffe2_evaluation(self):
      # Load onnx model with onnx module
      model = onnx.load(self.onnx_model)
      # For caffe2, it must be prepared a backed (kind of the "session" in tensorflow)
      prepared_backend = onnx_caffe2_backend.prepare(model, device="CPU")
      # Initialize the static graph
      W = {model.graph.input[0].name: self.x}
      # Inference
      caffe2_predictions = prepared_backend.run(W)[0]
      
      # Calculate accuracy
      score = Evaluation.score(y_true=self.y, y_pred=caffe2_predictions)
      print(f"Caffe2 accuracy: {score}")
      
   def tensorflow_evaluation(self):
      # Load the onnx model with onnx module
      model = onnx.load(self.onnx_model)
      # The "prepare" function is a wrapper for a session in tensorflow
      # Inference
      tensorflow_predictions = prepare(model).run(self.x)[0] 
      # Calculate accuracy
      score = Evaluation.score(y_true=self.y, y_pred=tensorflow_predictions)
      print(f"Tensorflow accuracy: {score}")