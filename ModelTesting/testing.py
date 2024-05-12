import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import onnxruntime as rt
import numpy as np

# Load the ONNX model
sess = rt.InferenceSession("mnist_conv_maxpool.onnx")

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the MNIST test data
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# Initialize counters
correct = 0
total = 0

# Iterate over all images in the test set
for images, labels in testloader:
    # Convert the image to a numpy array and add a batch dimension
    img_data = images.numpy()

    # Run inference with your model
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    result = sess.run([output_name], {input_name: img_data})

    # Get the predicted class (the index of the maximum value)
    _, predicted = torch.max(torch.from_numpy(result[0]), 1)

    # Update counters
    total += labels.size(0)
    print(predicted, labels)
    correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = correct / total
print('Accuracy of the model on the test images: {}%'.format(100 * accuracy))