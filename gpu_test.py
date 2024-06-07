import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPUs are available
if not torch.cuda.is_available():
    print("No GPUs are available.")
    exit()

# Check the number of GPUs available
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

# Ensure there are at least 4 GPUs
if num_gpus < 4:
    print("This script requires at least 4 GPUs.")
    exit()

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1000, 10)
    
    def forward(self, x):
        return self.fc(x)

#(.cuda moves to cuda)
# Create a model instance and move it to GPU
model = SimpleModel()
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model = model.cuda()

# Create a simple dataset
data = torch.randn(64, 1000).cuda()
target = torch.randint(0, 10, (64,)).cuda()

# Create loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Done training!")
