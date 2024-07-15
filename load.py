from torch import load, device, cuda
from torch.optim import SGD

from architectures import get_architecture

torch_device = device('cuda' if cuda.is_available() else 'cpu')
lr = 0.1
momentum = 0.9
weight_decay = 1e-4

model = get_architecture(torch_device)
optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

checkpoint = load('path_to_saved_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
