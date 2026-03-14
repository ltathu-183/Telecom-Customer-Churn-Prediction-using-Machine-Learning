# src/models/neural_network.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

class ChurnNeuralNetwork(nn.Module):
    """PyTorch neural network for churn prediction"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout_rate=0.3):
        super(ChurnNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class NeuralNetworkTrainer:
    """Train neural network with PyTorch"""
    
    def __init__(self, config=None):
        self.config = config or {
            'hidden_dims': [64, 32, 16],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10
        }
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        """Train neural network"""
        input_dim = X_train.shape[1]
        
        # Initialize model
        self.model = ChurnNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=self.config['hidden_dims'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        
        # Validation data
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'models/best_nn_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.config['epochs']}, "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.config['epochs']}, "
                          f"Train Loss: {train_loss:.4f}")
        
        # Load best model
        if X_val is not None:
            self.model.load_state_dict(torch.load('models/best_nn_model.pth'))
        
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict churn probabilities"""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy().flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (same as predict for binary classification)"""
        return self.predict(X)


class ModelEnsembler:
    """Combine multiple models for better performance"""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of predictions"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'predict'):
                pred = model.predict(X)
            else:
                raise ValueError("Model must have predict_proba or predict method")
            
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def predict(self, X: np.ndarray, threshold=0.5) -> np.ndarray:
        """Predict classes"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)