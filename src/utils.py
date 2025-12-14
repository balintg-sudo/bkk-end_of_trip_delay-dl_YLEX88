import sys
import logging
import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F

def get_logger(name):
    
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Teszt üzenet importáláskor
class BKK_GNN_Weighted(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(BKK_GNN_Weighted, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        input_dim = hidden_channels + 8 
        self.lin1 = Linear(input_dim, 64)
        self.lin2 = Linear(64, 32)
        self.lin3 = Linear(32, 1)

    def forward(self, x, edge_index, edge_weight, dynamic_features, current_stop_indices):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        
        batch_node_embeddings = x[current_stop_indices]
        
        combined = torch.cat([batch_node_embeddings, dynamic_features], dim=1)
        
        out = self.lin1(combined)
        out = out.relu()
        out = self.lin2(out)
        out = out.relu()
        out = self.lin3(out)
        return out