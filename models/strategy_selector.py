import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StrategySelector(nn.Module):
    """
    Neural network that selects strategies based on conversation state
    
    Input: Conversation state embedding (384-dim from sentence transformer)
    Output: Probability distribution over strategy combinations (5-dim, one per strategy)
    """
    
    def __init__(self, state_dim=384, num_strategies=5, hidden_dims=[128, 64]):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_strategies = num_strategies
        
        # Build network
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, num_strategies))
        
        self.network = nn.Sequential(*layers)
        
        print(f"Initialized StrategySelector:")
        print(f"  Input dim: {state_dim}")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Output dim: {num_strategies}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters())}")
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)
        
        Returns:
            Logits of shape (batch_size, num_strategies) or (num_strategies,)
        """
        return self.network(state)
    
    def select_strategies(self, state, threshold=0.5, temperature=1.0):
        """
        Select strategies based on state
        
        Args:
            state: numpy array of shape (state_dim,)
            threshold: Probability threshold for selecting a strategy
            temperature: Sampling temperature (lower = more confident)
        
        Returns:
            List of selected strategy indices
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            logits = self.forward(state_tensor) / temperature
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Select strategies above threshold
        selected = np.where(probs > threshold)[0].tolist()
        
        # If nothing selected, choose top strategy
        if not selected:
            selected = [np.argmax(probs)]
        
        return selected, probs
    
    def get_strategy_probs(self, state):
        """Get probability distribution over strategies"""
        with torch.no_grad():
            device = next(self.parameters()).device
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            logits = self.forward(state_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        return probs


# Test the model
if __name__ == '__main__':
    # Device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create model
    model = StrategySelector(state_dim=384, num_strategies=5).to(device)
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_state = torch.randn(1, 384).to(device)
    logits = model(dummy_state)
    print(f"Input shape: {dummy_state.shape}")
    print(f"Output shape: {logits.shape}")
    
    probs = torch.sigmoid(logits)
    print(f"Strategy probabilities: {probs.squeeze().detach().cpu().numpy()}")
    
    # Test strategy selection
    print("\nTesting strategy selection...")
    dummy_state_np = np.random.randn(384)
    selected, all_probs = model.select_strategies(dummy_state_np, threshold=0.5)
    
    strategies = ['empathy', 'validation', 'active_listening', 'problem_solving', 'authority']
    print(f"Selected strategy indices: {selected}")
    print(f"Selected strategies: {[strategies[i] for i in selected]}")
    print(f"All probabilities: {all_probs}")
    
    # Test batch processing
    print("\nTesting batch processing...")
    batch_states = torch.randn(16, 384).to(device)
    batch_logits = model(batch_states)
    print(f"Batch input shape: {batch_states.shape}")
    print(f"Batch output shape: {batch_logits.shape}")

