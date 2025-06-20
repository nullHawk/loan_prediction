#!/usr/bin/env python3
"""
Training script for Deep Loan Prediction Neural Network
Optimized for the best performing deep model architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

from model import (
    LoanPredictionDeepANN,
    load_processed_data,
    calculate_class_weights,
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix,
    model_summary
)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=2, gamma=2, logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

class DeepLoanTrainer:
    """Training pipeline for Deep Neural Network"""
    
    def __init__(self, learning_rate=0.012, batch_size=1536, device=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Using device: {self.device}")
        
        # Initialize model
        self.model = LoanPredictionDeepANN().to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def prepare_data(self, data_path='data/processed', validation_split=0.2):
        """Load and prepare data for training"""
        print("📊 Loading processed data...")
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
        
        # Create weighted sampler for imbalanced data
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Calculate class weights
        self.class_weights = calculate_class_weights(y_train)
        
        print(f"✅ Data preparation complete:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        print(f"   Test samples: {len(X_test):,}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Class weights: {self.class_weights}")
        
        return self
    
    def setup_training(self, weight_decay=1e-4):
        """Setup training configuration"""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # Loss function - Focal Loss for imbalanced data
        self.criterion = FocalLoss(alpha=2, gamma=2, logits=True)
        
        print("⚙️  Training setup complete:")
        print(f"   Optimizer: AdamW (lr={self.learning_rate}, weight_decay={weight_decay})")
        print(f"   Scheduler: CosineAnnealingWarmRestarts")
        print(f"   Loss: Focal Loss (alpha=2, gamma=2)")
        
        return self
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # Forward pass - model returns logits for deep ANN
            output = self.model(data)
            
            # Convert sigmoid output to logits for FocalLoss
            # Since DeepANN returns sigmoid output, convert to logits
            eps = 1e-7
            output_clamped = torch.clamp(output, eps, 1 - eps)
            logits = torch.log(output_clamped / (1 - output_clamped))
            
            loss = self.criterion(logits, target)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Predictions
            predicted = output > 0.5
            
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
                # Forward pass
                output = self.model(data)
                
                # Convert sigmoid output to logits for FocalLoss
                eps = 1e-7
                output_clamped = torch.clamp(output, eps, 1 - eps)
                logits = torch.log(output_clamped / (1 - output_clamped))
                
                loss = self.criterion(logits, target)
                
                predicted = output > 0.5
                
                total_loss += loss.item()
                total += target.size(0)
                correct += predicted.eq(target > 0.5).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs=200, early_stopping_patience=30, save_best=True):
        """Train the model"""
        print(f"\n🏋️  Starting training for {num_epochs} epochs...")
        print("=" * 80)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_accuracy = 0.0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
                lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch:3d}/{num_epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}% | '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}% | '
                      f'LR: {lr:.6f}')
            
            # Early stopping based on validation accuracy (for better performance)
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    self.save_model('best_deep_model.pth')
                    print(f"💾 New best model saved! Accuracy: {val_acc:.1f}%")
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience and epoch > 50:
                print(f"⏹️  Early stopping triggered after {epoch} epochs")
                break
        
        print("=" * 80)
        print("✅ Training completed!")
        
        # Load best model if saved
        if save_best and os.path.exists('best_deep_model.pth'):
            self.load_model('best_deep_model.pth')
            print("📥 Loaded best model weights.")
        
        return self
    
    def evaluate(self, threshold=0.5):
        """Evaluate the model on test set"""
        print("\n📈 Evaluating model on test set...")
        
        # Custom evaluation for DeepANN that returns sigmoid output
        self.model.eval()
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.X_test_np)
            y_pred_proba = self.model(X_test_tensor).numpy().flatten()
            y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(self.y_test_np, y_pred),
            'precision': precision_score(self.y_test_np, y_pred),
            'recall': recall_score(self.y_test_np, y_pred),
            'f1_score': f1_score(self.y_test_np, y_pred),
            'auc_roc': roc_auc_score(self.y_test_np, y_pred_proba)
        }
        
        print("\n📊 Test Set Performance:")
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
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'feature_names': self.feature_names
        }, filepath)
    
    def load_model(self, filepath):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history if available
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_accuracies = checkpoint['val_accuracies']
        
        print(f"✅ Model loaded from {filepath}")


def main():
    """Main training function"""
    print("🎯 Deep Loan Prediction Neural Network Training")
    print("=" * 60)
    
    # Configuration
    config = {
        'learning_rate': 0.012,         # Optimized learning rate
        'batch_size': 1536,             # Optimized batch size
        'num_epochs': 200,              # Sufficient epochs
        'early_stopping_patience': 30,  # Patience for early stopping
        'weight_decay': 1e-4,           # Regularization
        'validation_split': 0.2         # 20% for validation
    }
    
    print("⚙️  Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Initialize trainer
    trainer = DeepLoanTrainer(
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size']
    )
    
    # Show model architecture
    print("\n🏗️  Model Architecture:")
    model_summary(trainer.model)
    
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
    model_filename = f"loan_prediction_deep_model_{timestamp}.pth"
    trainer.save_model(model_filename)
    print(f"\n💾 Final model saved as: {model_filename}")
    
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
    
    results_filename = f"deep_training_results_{timestamp}.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Training results saved as: {results_filename}")
    
    # Performance Analysis
    print("\n" + "=" * 60)
    print("🎯 PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    final_accuracy = metrics['accuracy']
    if final_accuracy > 0.80:
        print(f"🏆 EXCELLENT: Accuracy of {final_accuracy:.1%} achieved!")
        print("    Outstanding performance for loan prediction!")
    elif final_accuracy > 0.70:
        print(f"✅ VERY GOOD: Accuracy of {final_accuracy:.1%} achieved!")
        print("    Great performance for this challenging problem!")
    elif final_accuracy > 0.60:
        print(f"👍 GOOD: Accuracy of {final_accuracy:.1%} achieved!")
        print("    Solid improvement over baseline!")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT: Accuracy of {final_accuracy:.1%}")
        print("    Consider additional optimization or feature engineering")
    
    print(f"\n📊 Key Metrics:")
    print(f"   • Accuracy: {metrics['accuracy']:.1%}")
    print(f"   • Precision: {metrics['precision']:.1%}")
    print(f"   • Recall: {metrics['recall']:.1%}")
    print(f"   • F1-Score: {metrics['f1_score']:.1%}")
    print(f"   • AUC-ROC: {metrics['auc_roc']:.3f}")
    
    # Business insights
    print(f"\n💼 Business Impact:")
    precision = metrics['precision']
    recall = metrics['recall']
    
    if precision > 0.85:
        print(f"   ✅ High Precision ({precision:.1%}): Low false positive rate")
        print(f"      → Minimizes bad loan approvals")
    if recall > 0.70:
        print(f"   ✅ Good Recall ({recall:.1%}): Catches most good applications")
        print(f"      → Maintains business volume")
    elif recall < 0.60:
        print(f"   ⚠️  Low Recall ({recall:.1%}): May reject too many good loans")
        print(f"      → Consider adjusting threshold")
    
    return trainer, metrics


if __name__ == "__main__":
    trainer, metrics = main()
    print(f"\n🎉 Training completed! Final accuracy: {metrics['accuracy']:.1%}")
    print("🚀 Model is ready for production use!")
