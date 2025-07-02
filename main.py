from data_process import ProcessedTemporalDataset
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN
import ssl
import numpy as np
from MA_TGCN import MATGCN
from torch_geometric.loader import DataLoader
ssl._create_default_https_context = ssl._create_unverified_context



dataset = ProcessedTemporalDataset(root="data/graph/")

train_size = 18000


train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]



class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, num_heads):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell with multi-head attention
        self.tgnn = MATGCN(in_channels=node_features,
                           out_channels=16,
                           periods=periods,
                           num_heads=num_heads)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(16, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h


#训练与测试
# GPU support
device = torch.device('cpu') # or cuda


# Create model and optimizers
model = TemporalGNN(node_features=1, periods=1, num_heads=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

print("Running training...")
y_pred1 = []
y_real1 = []

train_loader = DataLoader(train_dataset, batch_size=900, shuffle=True)

epoch_nums = 500

for epoch in range(epoch_nums):
    loss = 0
    step = 0
    for batch in train_loader:
        batch = batch.to(device)
        y_hat = model(batch.x, batch.edge_index)
        loss += torch.mean((y_hat - batch.y) ** 2)
        step += 1

    loss = loss / (step + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))


# test
model.eval()
loss = 0
step = 0


# Store for analysis
predictions = []
labels = []

y_pred = []
y_real = []

for snapshot in test_dataset:
    snapshot = snapshot.to(device)
    y_hat = model(snapshot.x, snapshot.edge_index)
    y_pred.append(y_hat.detach().cpu().numpy().flatten())
    y_real.append(snapshot.y.detach().cpu().numpy().flatten())

    # Mean squared error
    loss = loss + torch.mean((y_hat-snapshot.y)**2)
    # Store for analysis below
    labels.append(snapshot.y)
    predictions.append(y_hat)
    step += 1

y_pred = np.column_stack(y_pred)
y_real = np.column_stack(y_real)

loss = loss / (step+1)
loss = loss.item()
print("Test MSE: {:.4f}".format(loss))


np.save('result/y_pred.npy', y_pred)
np.save('result/y_real.npy', y_real)
