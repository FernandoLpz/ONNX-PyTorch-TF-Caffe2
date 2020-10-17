import onnx
import onnxruntime

class Evaluation:
   def __init__(self, onnx_model):
      self.onnx_model = onnx_model
      
   def torch_evaluation(self, x, y):
   	# with torch.no_grad():
   	#   model.eval()
   	#   ypred = model(x)
   	# return model.score(y, ypred)
   	pass
   
   def onnx_runtime_evaluation(self, x, y, print_graph=False):
      
      if print_graph:
      	# Load the ONNX model
      	model_onnx = onnx.load(self.onnx_model)
      	print(onnx.helper.printable_graph(model_onnx.graph))

      x = x.astype('float32')
      onnx_session = onnxruntime.InferenceSession(self.onnx_model)
      # compute ONNX Runtime output prediction
      onnx_inputs = {onnx_session.get_inputs()[0].name : x}
      onnx_outputs = onnx_session.run(None, onnx_inputs)
      print(f"onnx outputs: \n{onnx_outputs}")
      pass