import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNMoleculeModel(torch.nn.Module):
    def __init__(self, num_node_features, num_targets):
        super(GNNMoleculeModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.fc = torch.nn.Linear(128, num_targets)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        # Nie dajemy Sigmoid tutaj, użyjemy BCEWithLogitsLoss dla stabilności
        return x