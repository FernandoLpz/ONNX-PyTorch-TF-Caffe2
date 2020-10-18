import onnx
import onnxruntime

class Evaluation:
   def __init__(self, onnx_model):
      self.onnx_model = onnx_model
      
   @staticmethod
   def score(y_true, y_pred):
      tp, fp = 0, 0
      for ytrue, ypred in zip(y_true, y_pred):
         if (ytrue == 1) and (ypred >= 0.5):
            tp += 1
         elif (ytrue == 0) and (ypred < 0.5):
            fp += 1
      return (tp  + fp)/len(y_true)
		
   def torch_evaluation(self, x, y):
   	# with torch.no_grad():
   	#   model.eval()
   	#   ypred = model(x)
   	# return model.score(y, ypred)
   	pass
   
   def onnxruntime_evaluation(self, x, y, print_graph=False):
      
      if print_graph:
         # The onnx model is loaded and it is printed
         # in a human-readble format
      	model_onnx = onnx.load(self.onnx_model)
      	print(onnx.helper.printable_graph(model_onnx.graph))

      # In order to evaluate the trained model, it will be required
      # to fix the datatypes. As default, the numpy array saved the test set
      # as float64, however it is required to feed the model with a float32 datatype
      x = x.astype('float32')
            
      # Lauch the session
      onnx_session = onnxruntime.InferenceSession(self.onnx_model)
      # Assigns the input numpy array as input vector for the onnx session
      onnx_inputs = {onnx_session.get_inputs()[0].name : x}
      # The method "run" evaluates the loaded model
      onnx_predictions = onnx_session.run(None, onnx_inputs)
      
      # Calculate accuracy
      score = Evaluation.score(y_true=y, y_pred=onnx_predictions[0])
      print(f"ONNX accuracy: {score}")
      
      pass