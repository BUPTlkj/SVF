import numpy as np
import onnxruntime as rt
import torch

# Load CSV data
data = np.loadtxt('mnist_test.csv', delimiter=',')

# Load ONNX model
sess = rt.InferenceSession("mnist_conv_maxpool.onnx")

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name

# Iterate over each row in the data
for i in range(data.shape[0]):
    # Slice the row to exclude the label and reshape to 28x28
    # Add two dimensions for batch and channels
    input_data = data[i, 1:].reshape(1, 1, 28, 28).astype(np.float32)

    # Run the model
    result = sess.run(None, {input_name: input_data})

    _, predicted = torch.max(torch.from_numpy(result[0]), 1)

    print(predicted)

    # print(result)