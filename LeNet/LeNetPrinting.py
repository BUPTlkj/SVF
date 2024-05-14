import numpy as np
import onnxruntime as rt
import torch

# Load CSV data
data = np.loadtxt('mnist_test.csv', delimiter=',')

# Load ONNX model
session = rt.InferenceSession("/Users/den184/Documents/UNSW/SVF/SVF-Kane/LeNet/save_model_original_MaxPool2d/best_model_MaxPool2d.onnx")

# Get the input name for the ONNX model
input_name = session.get_inputs()[0].name

# # Iterate over each row in the data
# for i in range(data.shape[0]):
#     # Slice the row to exclude the label and reshape to 28x28
#     # Add two dimensions for batch and channels


input_data = data[0, 1:].reshape(1, 1, 28, 28).astype(np.float32)





# # Get the names of all the layers in the model
# layer_names = [node.name for node in session.get_graph_info().node_arg]

# # Run the model on the input data and get the output from each layer
# outputs = session.run(layer_names, {"input": input_data})

# # Print the output from each layer
# for name, output in zip(layer_names, outputs):
#     print(f"Output from {name}: {output}")
