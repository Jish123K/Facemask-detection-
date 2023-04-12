# Facemask-detection-
This code first defines a PyTorch model with the same architecture as the Keras model, loads the model's parameters from a PyTorch checkpoint file (maskvsnomask.pth), and moves the model to the GPU if it's available.

Then, it loads the input image using the PIL library, applies a series of transforms to resize and convert the image to a PyTorch tensor, and moves the tensor to the GPU.

Finally, it passes the tensor through the model to obtain a prediction, and prints the result
