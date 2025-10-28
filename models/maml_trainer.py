import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
import pickle
from models.strategy_selector import StrategySelector

class MAMLTrainer:
    """
    Model-Agnostic Meta-Learning trainer for strategy selection
    
    Based on Finn et al. 2017 and Madotto et al. 2019
    
    Key insight: Train a model that can quickly adapt to new personas
    with just a few examples (few-shot learning).
    """
    
    def __init__(
        self,
        model,
        inner_lr=0.01,
        outer_lr=0.001,
        num_inner_steps=5,
        device='cpu'
    ):
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.device = device
        
        # Meta-optimizer (updates base model)
        self.meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
        
        print(f"Initialized MAML Trainer:")
        print(f"  Inner LR: {inner_lr}")
        print(f"  Outer LR: {outer_lr}")
        print(f"  Inner steps: {num_inner_steps}")
        print(f"  Device: {device}")
    
    def train_step(self, task_batch):
        """
        Single MAML training step
        
        Args:
            task_batch: List of tasks, each task is dict with 'data' field
        
        Returns:
            meta_loss: Float, loss across all tasks
        """
        meta_loss = 0.0
        num_valid_tasks = 0
        
        for task in task_batch:
            # Split task into support (adaptation) and query (evaluation)
            support_data = task['data'][:3]  # First 3 turns for adaptation
            query_data = task['data'][3:]     # Remaining turns for evaluation
            
            if not query_data:  # Need at least one query point
                continue
            
            # Inner loop: Adapt to this task using support data
            # This computes adapted parameters while maintaining gradients
            adapted_params = self._inner_loop(support_data)
            
            # Outer loop: Evaluate adapted model on query data
            task_loss = self._compute_loss_with_params(adapted_params, query_data)
            meta_loss += task_loss
            num_valid_tasks += 1
        
        if num_valid_tasks == 0:
            return 0.0
        
        # Meta-update: Update base model to improve adaptation
        meta_loss = meta_loss / num_valid_tasks
        
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def _inner_loop(self, support_data, create_graph=True):
        """
        Inner loop: Fast adaptation to new task
        
        This is the key to MAML - we compute adapted parameters
        while maintaining the computational graph for meta-learning.
        
        Args:
            support_data: List of (state, action, reward) dicts
            create_graph: Whether to create computation graph (True for training, False for eval)
        
        Returns:
            adapted_params: Dictionary of adapted parameters
        """
        # Start with current model parameters
        # CRITICAL: Clone and explicitly set requires_grad to maintain gradients
        params = {
            name: param.clone().requires_grad_(True) 
            for name, param in self.model.named_parameters()
        }
        
        # Adapt for N inner steps using gradient descent
        for step in range(self.num_inner_steps):
            loss = self._compute_loss_with_params(params, support_data)
            
            # Compute gradients with respect to current params
            grads = torch.autograd.grad(
                loss,
                params.values(),
                create_graph=create_graph  # Maintain graph for meta-learning
            )
            
            # Update parameters using gradient descent
            params = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(params.items(), grads)
            }
        
        return params
    
    def _compute_loss_with_params(self, params, data):
        """
        Compute loss using specific parameters (functional approach)
        
        Args:
            params: Dictionary of parameters
            data: List of (state, action, reward) dicts
        
        Returns:
            loss: Scalar tensor
        """
        total_loss = 0.0
        
        for turn in data:
            state = torch.tensor(turn['state'], dtype=torch.float32).to(self.device)
            action = torch.tensor(turn['action'], dtype=torch.float32).to(self.device)
            reward = turn['reward']
            
            # Forward pass using functional API
            logits = self._forward_with_params(state, params)
            
            # Binary cross-entropy loss (multi-label classification)
            # Learn to predict which strategies were used
            loss = F.binary_cross_entropy_with_logits(
                logits,
                action,
                reduction='mean'
            )
            
            # Weight by reward magnitude (optional: weight important examples more)
            # Use absolute value to avoid negative losses
            # Add 1.0 to ensure all examples contribute
            reward_weight = 1.0 + abs(reward)
            weighted_loss = loss * reward_weight
            
            total_loss += weighted_loss
        
        return total_loss / len(data)
    
    def _forward_with_params(self, state, params):
        """
        Forward pass through network using specific parameters
        
        This implements the network forward pass functionally,
        which is needed for MAML's higher-order gradients.
        """
        x = state
        
        # Manually traverse the network using provided parameters
        # Assumes network structure: Linear -> ReLU -> Dropout -> ... -> Linear
        param_list = list(params.values())
        param_idx = 0
        
        for layer in self.model.network:
            if isinstance(layer, nn.Linear):
                weight = param_list[param_idx]
                bias = param_list[param_idx + 1]
                x = F.linear(x, weight, bias)
                param_idx += 2
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x)
            elif isinstance(layer, nn.Dropout):
                # During training, keep dropout; during eval, skip it
                if self.model.training:
                    x = F.dropout(x, p=layer.p, training=True)
        
        return x
    
    def adapt_to_persona(self, support_data):
        """
        Adapt model to a new persona using support data
        
        Args:
            support_data: List of (state, action, reward) dicts
        
        Returns:
            adapted_params: Adapted parameters for this persona
        """
        self.model.eval()
        # Need gradients for adaptation, but don't need graph
        adapted_params = self._inner_loop(support_data, create_graph=False)
        return adapted_params
    
    def predict_with_params(self, params, state):
        """
        Make prediction using specific parameters
        
        Args:
            params: Dictionary of adapted parameters
            state: Numpy array of shape (state_dim,)
        
        Returns:
            probs: Numpy array of strategy probabilities
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self._forward_with_params(state_tensor, params)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs
    
    def evaluate(self, task_batch):
        """
        Evaluate model on tasks without updating
        
        Returns:
            avg_loss: Average loss across tasks
        """
        self.model.eval()
        total_loss = 0.0
        num_valid_tasks = 0
        
        # Don't use torch.no_grad() here because _inner_loop needs gradients
        for task in task_batch:
            support_data = task['data'][:3]
            query_data = task['data'][3:]
            
            if not query_data:
                continue
            
            # Adapt and evaluate (without creating graph for efficiency)
            adapted_params = self._inner_loop(support_data, create_graph=False)
            
            # Compute loss without gradients
            with torch.no_grad():
                task_loss = self._compute_loss_with_params(adapted_params, query_data)
                total_loss += task_loss.item()
                num_valid_tasks += 1
        
        return total_loss / num_valid_tasks if num_valid_tasks > 0 else 0.0


def train_maml(tasks_path='data/maml_tasks.pkl', num_epochs=100, batch_size=5):
    """
    Train MAML model
    
    Args:
        tasks_path: Path to pickled tasks
        num_epochs: Number of meta-training epochs
        batch_size: Number of tasks per batch
    """
    # Load tasks
    with open(tasks_path, 'rb') as f:
        all_tasks = pickle.load(f)
    
    print(f"Loaded {len(all_tasks)} tasks")
    
    # Split into train/val
    np.random.shuffle(all_tasks)
    split_idx = int(0.8 * len(all_tasks))
    train_tasks = all_tasks[:split_idx]
    val_tasks = all_tasks[split_idx:]
    
    print(f"Train tasks: {len(train_tasks)}")
    print(f"Val tasks: {len(val_tasks)}")
    
    # Initialize model and trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StrategySelector(state_dim=384, num_strategies=5)
    trainer = MAMLTrainer(
        model,
        inner_lr=0.01,        # Keep moderate for adaptation
        outer_lr=0.0005,      # REDUCED from 0.001 - slower meta-learning
        num_inner_steps=3,    # REDUCED from 5 - less overfitting to support set
        device=device
    )
    
    # Training loop with early stopping
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5  # Stop if no improvement for 5 validation checks
    patience_counter = 0
    best_model_state = None
    
    print("\nStarting MAML training...\n")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        np.random.shuffle(train_tasks)
        epoch_train_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_tasks), batch_size):
            batch = train_tasks[i:i+batch_size]
            loss = trainer.train_step(batch)
            epoch_train_loss += loss
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)
        
        # Validation every 5 epochs (more frequent than before)
        if epoch % 5 == 0:
            val_loss = trainer.evaluate(val_tasks)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  → New best validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} checks)")
                    break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with val loss: {best_val_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'results/maml_model.pt')
    print("\nModel saved to results/maml_model.pt")
    
    # Save training curves
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    
    if val_losses:
        val_epochs = list(range(0, len(train_losses), 5))[:len(val_losses)]
        plt.plot(val_epochs, val_losses, label='Val Loss', marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Meta Loss')
    plt.title('MAML Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add best val loss marker
    if best_model_state is not None:
        best_epoch = val_epochs[val_losses.index(best_val_loss)]
        plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best model (epoch {best_epoch})')
        plt.legend()
    
    plt.savefig('results/maml_training_curve.png', dpi=300, bbox_inches='tight')
    print("Training curve saved to results/maml_training_curve.png")
    
    return model, train_losses, val_losses


if __name__ == '__main__':
    model, train_losses, val_losses = train_maml(
        num_epochs=100,
        batch_size=5
    )

