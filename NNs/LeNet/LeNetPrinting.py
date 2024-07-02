import numpy as np
import onnxruntime as rt
import torch
import onnx


# Load CSV data
data = np.loadtxt('mnist_test.csv', delimiter=',')

path = "/Users/den184/Documents/UNSW/SVF/SVF-Kane/LeNet/save_model_original_MaxPool2d/best_model_MaxPool2d.onnx"

# Load ONNX model
model = model = onnx.load(path)
# Inference ready...
session = rt.InferenceSession(path)



# Get the input name for the ONNX model
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

print(input_name)
print(output_names)

input_data = data[0, 1:].reshape(1, 1, 28, 28).astype(np.float32)

# # Get the names of all the layers in the model
# layer_names = [node.name for node in session.get_graph_info().node_arg]

# # Run the model on the input data and get the output from each layer
outputs = session.run(output_names, {"input": input_data})

# Print the output from each layer
for name, output in zip(output_names, outputs):
    print(f"Output from {name}: {output}")
