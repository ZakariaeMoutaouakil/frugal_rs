# Replace 'yourfile.h5' with the path to your file
import h5py

file = h5py.File('/home/pc/Projects/private_data/test_results/cifar10_0.50.h5', 'r')
predictions = file['cifar10_0.5_predictions'][0]

print(predictions)
