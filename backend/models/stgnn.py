import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Tuple

class STGNNModel(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for probabilistic traffic forecasting.
    Using simple Graph Convolutional Networks (GCN) combined with GRU to model
    both structural network dependencies and temporal variations over the road graph.
    """
    def __init__(self, node_features: int, hidden_dim: int, output_dim: int):
        super(STGNNModel, self).__init__()
        
        # Spatial Processing Layers
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Temporal Sequence Processing Layer
        # Processes sequences of spatial embeddings over time. Note: batch_first=True
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        # Output Layer
        # We output a mean (mu) and a standard deviation (sigma) for the probabilistic prediction
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            x: shape [num_nodes, seq_len, num_features]
            edge_index: PyTorch Geometric standard, [2, num_edges]
            edge_weight: shape [num_edges] or [num_edges, 1]
            
        Returns:
            mu: predicted average congestion per node [num_nodes]
            sigma: standard deviation of congestion [num_nodes]
        """
        num_nodes, seq_len, _ = x.shape
        out_seq = []
        
        for t in range(seq_len):
            # Process each timestep spatially
            xt = x[:, t, :]
            
            # 1st GCN Layer
            xt = self.conv1(xt, edge_index, edge_weight)
            xt = F.relu(xt)
            
            # 2nd GCN Layer
            xt = self.conv2(xt, edge_index, edge_weight)
            xt = F.relu(xt)
            
            out_seq.append(xt)
            
        # Re-stack temporal sequence: [num_nodes, seq_len, hidden_dim]
        out_seq = torch.stack(out_seq, dim=1)
        
        # Temporal Modeling via GRU
        out, _ = self.gru(out_seq)

        # Grab the last timestep's output for forecasting
        last_out = out[:, -1, :] # [num_nodes, hidden_dim]
        
        preds = self.fc(last_out) # [num_nodes, 2]
        
        # Extract probabilistic mean and scale
        mu = torch.sigmoid(preds[:, 0]) # Constrain relative congestion to [0,1]
        sigma = F.softplus(preds[:, 1]) + 1e-6 # Ensure strictly positive std deviation

        return mu, sigma
