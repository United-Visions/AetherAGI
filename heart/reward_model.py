"""
Path: heart/reward_model.py
Role: 2-layer MLP to predict human flourishing.
"""
import torch
import torch.nn as nn
from loguru import logger

class RewardModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128):
        """
        A simple 2-layer MLP (Multi-Layer Perceptron) to act as the reward model.
        It takes a state vector and predicts a 'human flourishing' score.
        """
        super(RewardModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh() # To ensure output is between -1 and 1
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        logger.info("RewardModel (2-layer MLP) initialized.")

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict the flourishing score.
        """
        x = self.layer1(state_vector)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.tanh(x)
        return x

    def predict_flourishing(self, state_vector: list) -> float:
        """
        Takes a state vector, converts to tensor, and predicts flourishing.
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            tensor_in = torch.FloatTensor(state_vector).unsqueeze(0)
            prediction = self.forward(tensor_in)
            return prediction.item()

    def update_model(self, state_vector: list, actual_score: float):
        """
        Performs one step of backpropagation to train the model.
        """
        self.train() # Set model to training mode
        
        tensor_in = torch.FloatTensor(state_vector).unsqueeze(0)
        target = torch.FloatTensor([actual_score])

        # Forward pass
        prediction = self.forward(tensor_in)
        
        # Compute loss
        loss = self.loss_fn(prediction.squeeze(), target)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        logger.info(f"RewardModel updated. Loss: {loss.item():.4f}")

def save_model(model, path="heart/weights.pt"):
    torch.save(model.state_dict(), path)
    logger.info(f"RewardModel saved to {path}")

def load_model(path="heart/weights.pt", input_dim=1024):
    model = RewardModel(input_dim)
    try:
        model.load_state_dict(torch.load(path))
        logger.info(f"RewardModel loaded from {path}")
    except FileNotFoundError:
        logger.warning(f"No weights file found at {path}. Initializing with random weights.")
    return model
