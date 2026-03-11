import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from loguru import logger
from datetime import datetime

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.stgnn import STGNNModel
from backend.services.graph_builder import GraphBuilder

def generate_synthetic_data(num_nodes: int, seq_len: int, num_features: int, num_samples: int = 100):
    """
    Generates synthetic sequential data for ST-GNN training.
    X shape: [num_samples, num_nodes, seq_len, num_features]
    Y shape: [num_samples, num_nodes] (Target congestion [0, 1])
    """
    X = torch.rand((num_samples, num_nodes, seq_len, num_features), dtype=torch.float32)
    # Simple target: moving average of the last feature across timesteps + some noise
    Y = X.mean(dim=2).mean(dim=-1) + torch.randn(num_samples, num_nodes) * 0.05
    Y = torch.clamp(Y, 0.0, 1.0)
    return X, Y

def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative Log-Likelihood for Gaussian distribution.
    Assumes prediction outputs mean (mu) and standard deviation (sigma).
    """
    # NLL = 0.5 * log(sigma^2) + 0.5 * (target - mu)^2 / sigma^2
    var = sigma ** 2
    loss = 0.5 * torch.log(var) + 0.5 * ((target - mu) ** 2) / var
    return loss.mean()

def train_pipeline(epochs: int = 50, batch_size: int = 16):
    logger.info("Initializing GraphBuilder and retrieving graph structure...")
    builder = GraphBuilder(place_name="Bandra, Mumbai, India", use_cache=True)
    
    # We just need the graph structure (edge_index, edge_attr) and node count
    data, idx_node_map = builder.get_pytorch_geometric_data()
    num_nodes = data.x.shape[0]
    
    edge_index = data.edge_index
    edge_weight = data.edge_attr.squeeze(-1) # STGNNModel accepts [num_edges]
    
    logger.info(f"Graph loaded. Nodes: {num_nodes}, Edges: {edge_index.shape[1]}")
    
    seq_len = 12 # E.g., past 12 timesteps (1 hour if 5 min intervals)
    num_features = 2 # e.g., speed, occupancy
    hidden_dim = 32
    output_dim = 2 # mu, sigma
    
    logger.info("Instantiating STGNN Model...")
    model = STGNNModel(node_features=num_features, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    logger.info("Generating synthetic training data...")
    num_samples = 200
    X_train, Y_train = generate_synthetic_data(num_nodes, seq_len, num_features, num_samples)
    
    logger.info("Starting training loop...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Creating simple batches
        permutation = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_train[indices] # [batch, nodes, seq, feats]
            batch_Y = Y_train[indices] # [batch, nodes]
            
            # STGNN forward expects [nodes, seq_len, feats] for single graph instance
            # Since our model doesn't natively support batch dimension in graph (unless we use PyG Batching)
            # We will just accumulate gradients across the batch for simplicity.
            batch_loss = 0.0
            
            for b in range(len(indices)):
                x_b = batch_X[b] # [nodes, seq, feats]
                y_b = batch_Y[b] # [nodes]
                
                optimizer.zero_grad()
                mu, sigma = model(x_b, edge_index, edge_weight)
                loss = gaussian_nll_loss(mu, sigma, y_b)
                
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
                
            epoch_loss += batch_loss / len(indices)
            
        avg_epoch_loss = epoch_loss / (num_samples / batch_size)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}")
            
    # Save the trained weights
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "stgnn_weights.pt")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'node_features': num_features,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'seq_len': seq_len,
        'nodes_count': num_nodes
    }, save_path)
    
    logger.info(f"Training complete. Weights saved to {save_path}")
    return save_path

if __name__ == "__main__":
    train_pipeline(epochs=50)
