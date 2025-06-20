import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

from model import (
    LoanPredictionANN, 
    LoanPredictionLightANN, 
    LoanPredictionDeepANN,
    load_processed_data,
    calculate_class_weights,
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix,
    model_summary
)

class LoanPredictionTrainer:
    """
    Comprehensive trainer for Loan Prediction Neural Networks
    """
    
    def __init__(self, model_type='standard', learning_rate=0.001, batch_size=512, 
                 device=None, use_class_weights=True):
        """
        Initialize the trainer
        
        Args:
            model_type: 'light', 'standard', or 'deep'
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            device: Device to use ('cuda' or 'cpu')
            use_class_weights: Whether to use class weights for imbalanced data
        """
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_class_weights = use_class_weights
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def _create_model(self):
        """Create model based on specified type"""
        if self.model_type == 'light':
            return LoanPredictionLightANN()
        elif self.model_type == 'standard':
            return LoanPredictionANN()
        elif self.model_type == 'deep':
            return LoanPredictionDeepANN()
        else:
            raise ValueError("model_type must be 'light', 'standard', or 'deep'")
    
    def prepare_data(self, data_path='data/processed', validation_split=0.2):
        """Load and prepare data for training"""
        print("Loading processed data...")
        X_train, y_train, X_test, y_test, feature_names = load_processed_data(data_path)
        
        # Split training data into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, 
            random_state=42, stratify=y_train
        )
        
        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        self.X_val = torch.FloatTensor(X_val).to(self.device)
        self.y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_test = torch.FloatTensor(y_test).unsqueeze(1).to(self.device)
        
        # Store original numpy arrays for evaluation
        self.X_test_np = X_test
        self.y_test_np = y_test
        
        self.feature_names = feature_names
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Calculate class weights if needed
        if self.use_class_weights:
            self.class_weights = calculate_class_weights(y_train)
            print(f"Class weights: {self.class_weights}")
        else:
            self.class_weights = None
        
        print(f"Data prepared:")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Validation samples: {len(X_val):,}")
        print(f"  Test samples: {len(X_test):,}")
        print(f"  Features: {len(feature_names)}")
        
        return self
    
    def setup_training(self, weight_decay=1e-5):
        """Setup optimizer and loss function"""
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Loss function
        if self.use_class_weights and self.class_weights is not None:
            # Weighted BCE loss for imbalanced data
            pos_weight = self.class_weights[1] / self.class_weights[0]
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        else:
            self.criterion = nn.BCELoss()
        
        print(f"Training setup complete:")
        print(f"  Optimizer: Adam (lr={self.learning_rate}, weight_decay={weight_decay})")
        print(f"  Scheduler: ReduceLROnPlateau")
        print(f"  Loss function: {'Weighted BCE' if self.use_class_weights else 'BCE'}")
        
        return self
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            output = self.model(data)
            
            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                # Remove sigmoid from model output for BCEWithLogitsLoss
                output_logits = output  # Assuming output is logits
                loss = self.criterion(output_logits, target)
                predicted = torch.sigmoid(output_logits) > 0.5
            else:
                loss = self.criterion(output, target)
                predicted = output > 0.5
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total += target.size(0)
            correct += predicted.eq(target > 0.5).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                
                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    output_logits = output
                    loss = self.criterion(output_logits, target)
                    predicted = torch.sigmoid(output_logits) > 0.5
                else:
                    loss = self.criterion(output, target)
                    predicted = output > 0.5
                
                total_loss += loss.item()
                total += target.size(0)
                correct += predicted.eq(target > 0.5).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs=100, early_stopping_patience=20, save_best=True):
        """Train the model"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            if epoch % 10 == 0 or epoch == 1:
                print(f'Epoch {epoch:3d}/{num_epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    self.save_model('best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        print("=" * 60)
        print("Training completed!")
        
        # Load best model if saved
        if save_best and os.path.exists('best_model.pth'):
            self.load_model('best_model.pth')
            print("Loaded best model weights.")
        
        return self
    
    def evaluate(self, threshold=0.5):
        """Evaluate the model on test set"""
        print("\nEvaluating model on test set...")
        
        metrics, y_pred, y_pred_proba = evaluate_model(
            self.model, self.X_test_np, self.y_test_np, threshold
        )
        
        print("\nTest Set Performance:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Plot confusion matrix
        cm = plot_confusion_matrix(self.y_test_np, y_pred)
        
        # Plot training history
        plot_training_history(
            self.train_losses, self.val_losses, 
            self.train_accuracies, self.val_accuracies
        )
        
        return metrics, y_pred, y_pred_proba
    
    def save_model(self, filepath):
        """Save model and training state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'feature_names': self.feature_names
        }, filepath)
    
    def load_model(self, filepath):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history if available
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_accuracies = checkpoint['val_accuracies']
        
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """Print model summary"""
        model_summary(self.model)


def main():
    """Main training function"""
    print("Loan Prediction Neural Network Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'model_type': 'standard',  # 'light', 'standard', 'deep'
        'learning_rate': 0.001,
        'batch_size': 512,
        'num_epochs': 100,
        'weight_decay': 1e-5,
        'early_stopping_patience': 20,
        'use_class_weights': True,
        'validation_split': 0.2
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = LoanPredictionTrainer(
        model_type=config['model_type'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        use_class_weights=config['use_class_weights']
    )
    
    # Show model architecture
    trainer.get_model_summary()
    
    # Prepare data and setup training
    trainer.prepare_data(validation_split=config['validation_split'])
    trainer.setup_training(weight_decay=config['weight_decay'])
    
    # Train the model
    trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    # Evaluate the model
    metrics, predictions, probabilities = trainer.evaluate()
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"loan_prediction_model_{config['model_type']}_{timestamp}.pth"
    trainer.save_model(model_filename)
    print(f"\nFinal model saved as: {model_filename}")
    
    # Save training results
    results = {
        'config': config,
        'final_metrics': metrics,
        'training_history': {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'train_accuracies': trainer.train_accuracies,
            'val_accuracies': trainer.val_accuracies
        }
    }
    
    results_filename = f"training_results_{timestamp}.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Training results saved as: {results_filename}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
