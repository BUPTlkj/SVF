import onnx
import numpy as np

# Load the model
# model = onnx.load('/Users/den184/Documents/UNSW/SVF/test/SVF-Kane/NNs/SmallNet/CNN/No_MaxPool/save_model_CNN_No_MaxP/best_model_CNN_No_MaxP.onnx')

# # Iterate over the nodes
# for node in model.graph.node:
#     # Iterate over the attributes of the node
#     for attr in node.attribute:
#         # Check if the attribute is a tensor
#         if node.op_type == 'Conv':
#             if attr.t:
#                 tensor = np.array(attr.t.float_data)
#                 # Convert from NCHW to NHWC
#                 tensor_nhwc = np.transpose(tensor, (0, 2, 3, 1))
#                 print(tensor_nhwc)

# # model.graph.node[5].attribute[0]

import onnx
from onnx import numpy_helper
 
# Function to load an ONNX model and print conv filters transposed
def print_conv_filters_transposed(onnx_path):
    # Load the ONNX model
    model = onnx.load(onnx_path)
 
# Iterate through all the nodes in the graph
    for node in model.graph.node:
        # Check if the node is a convolution node
        if node.op_type == "Conv":
            # Iterate through the inputs of the conv node
            for input_name in node.input:
                # Find the corresponding initializer
                for initializer in model.graph.initializer:
                    if initializer.name == input_name:
                        # Convert the initializer to a numpy array
                        tensor = numpy_helper.to_array(initializer)
                        # Check if tensor has four dimensions
                        if tensor.ndim == 4:
                            # Transpose the tensor from (0,2,3,1)
                            transposed_tensor = tensor.transpose(0,2,3,1)
                            print(f"Filter for {node.name} (pre-transposing):")
                            print(tensor)

                            print(f"Filter for {node.name} (transposed):")
                            print(transposed_tensor)
                            print()
                        else:
                            print(f"Filter for {node.name} has unexpected shape: {tensor.shape}. Skipping transpose.")
 
# Example usage
# Replace 'your_model.onnx' with the path to your actual ONNX model file
print_conv_filters_transposed('/Users/den184/Documents/UNSW/SVF/test/SVF-Kane/NNs/SmallNet/CNN/No_MaxPool/save_model_CNN_No_MaxP/best_model_CNN_No_MaxP.onnx')
