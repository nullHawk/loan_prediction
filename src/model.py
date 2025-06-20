import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class LoanPredictionDeepANN(nn.Module):
    """
    Deeper version for maximum performance
    
    Architecture:
    - Input: 9 features
    - Hidden Layer 1: 128 neurons (ReLU)
    - Hidden Layer 2: 64 neurons (ReLU)
    - Hidden Layer 3: 32 neurons (ReLU)
    - Hidden Layer 4: 16 neurons (ReLU)
    - Output: 1 neuron (Sigmoid)
    - Dropout: [0.3, 0.3, 0.2, 0.1]
    """
    
    def __init__(self, input_size=9):
        super(LoanPredictionDeepANN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, 16)
        self.dropout4 = nn.Dropout(0.1)
        
        self.fc5 = nn.Linear(16, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        
        x = torch.sigmoid(self.fc5(x))
        
        return x


def load_processed_data(data_path='data/processed'):
    """Load the processed training and test data"""
    train_data = pd.read_csv(f'{data_path}/train_data_scaled.csv')
    test_data = pd.read_csv(f'{data_path}/test_data_scaled.csv')
    
    # Separate features and target
    feature_columns = [col for col in train_data.columns if col != 'loan_repaid']
    
    X_train = train_data[feature_columns].values
    y_train = train_data['loan_repaid'].values
    
    X_test = test_data[feature_columns].values
    y_test = test_data['loan_repaid'].values
    
    return X_train, y_train, X_test, y_test, feature_columns


def calculate_class_weights(y):
    """Calculate class weights for handling imbalanced data"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return torch.FloatTensor(weights)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Comprehensive model evaluation - updated for logits output"""
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_logits = model(X_test_tensor)
        y_pred_proba = torch.sigmoid(y_logits).numpy().flatten()
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc
    }
    
    return metrics, y_pred, y_pred_proba


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=['Charged Off', 'Fully Paid']):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return cm


def model_summary(model):
    """Print model architecture summary"""
    print("=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\nLayer Details:")
    print("-" * 40)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"{name}: {module}")
        elif isinstance(module, nn.Dropout):
            print(f"{name}: {module}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    print("Loading processed data...")
    X_train, y_train, X_test, y_test, feature_names = load_processed_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Feature names: {feature_names}")
    
    # Create model
    model = LoanPredictionDeepANN()
    model_summary(model)
    
    print("\nModel created successfully!")
    print("Use train.py to train the model.")