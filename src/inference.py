"""
Loan Prediction Inference Script

This script provides inference functionality for the trained loan prediction model.
It can handle both single predictions and batch predictions for loan approval decisions.

Usage:
    python inference.py --help                                    # Show help
    python inference.py --single                                  # Interactive single prediction
    python inference.py --batch input.csv output.csv            # Batch prediction
    python inference.py --sample                                  # Run with sample data
"""

import torch
import pandas as pd
import numpy as np
import json
import argparse
import sys
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import the model
from model import LoanPredictionDeepANN

class LoanPredictor:
    """
    Loan Prediction Inference Class
    
    This class handles loading the trained model, preprocessing input data,
    and making predictions for loan approval decisions.
    """
    
    def __init__(self, model_path='bin/best_checkpoint.pth', 
                 preprocessing_info_path='data/processed/preprocessing_info.json',
                 scaler_params_path='data/processed/scaler_params.csv'):
        """
        Initialize the LoanPredictor
        
        Args:
            model_path (str): Path to the trained model checkpoint
            preprocessing_info_path (str): Path to preprocessing configuration
            scaler_params_path (str): Path to scaler parameters
        """
        self.model_path = model_path
        self.preprocessing_info_path = preprocessing_info_path
        self.scaler_params_path = scaler_params_path
        
        # Initialize components
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.preprocessing_info = None
        
        # Load everything
        self._load_preprocessing_info()
        self._load_scaler()
        self._load_model()
        
        print("✅ LoanPredictor initialized successfully!")
        print(f"📊 Model expects {len(self.feature_names)} features")
        print(f"🎯 Features: {', '.join(self.feature_names)}")
    
    def _load_preprocessing_info(self):
        """Load preprocessing information"""
        try:
            with open(self.preprocessing_info_path, 'r') as f:
                self.preprocessing_info = json.load(f)
            
            # Define feature names based on the model
            self.feature_names = [
                'dti', 'credit_history_length', 'debt_to_credit_ratio',
                'revol_bal', 'installment', 'revol_util', 
                'int_rate', 'annual_inc', 'total_credit_lines'
            ]
            
            print(f"✅ Loaded preprocessing info from {self.preprocessing_info_path}")
            
        except Exception as e:
            print(f"❌ Error loading preprocessing info: {str(e)}")
            raise
    
    def _load_scaler(self):
        """Load and reconstruct the scaler from saved parameters"""
        try:
            scaler_params = pd.read_csv(self.scaler_params_path)
            
            # Reconstruct StandardScaler
            self.scaler = StandardScaler()
            self.scaler.mean_ = scaler_params['mean'].values
            self.scaler.scale_ = scaler_params['scale'].values
            # Calculate variance from scale (variance = scale^2)
            self.scaler.var_ = (scaler_params['scale'].values) ** 2
            self.scaler.n_features_in_ = len(scaler_params)
            self.scaler.feature_names_in_ = scaler_params['feature'].values
            
            print(f"✅ Loaded scaler parameters from {self.scaler_params_path}")
            
        except Exception as e:
            print(f"❌ Error loading scaler: {str(e)}")
            raise
    
    def _load_model(self):
        """Load the trained model"""
        try:
            # Initialize model architecture
            self.model = LoanPredictionDeepANN(input_size=len(self.feature_names))
            
            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set to evaluation mode
            self.model.eval()
            
            print(f"✅ Loaded model from {self.model_path}")
            print(f"📈 Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def preprocess_input(self, data):
        """
        Preprocess input data for prediction
        
        Args:
            data (dict or pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Preprocessed and scaled data
        """
        try:
            # Convert to DataFrame if dict
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError("Input data must be dict or DataFrame")
            
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Select and order features correctly
            df = df[self.feature_names]
            
            # Apply scaling
            scaled_data = self.scaler.transform(df.values)
            
            return scaled_data
            
        except Exception as e:
            print(f"❌ Error preprocessing data: {str(e)}")
            raise
    
    def predict_single(self, data, return_proba=True):
        """
        Make prediction for a single loan application
        
        Args:
            data (dict): Single loan application data
            return_proba (bool): Whether to return probability scores
            
        Returns:
            dict: Prediction results
        """
        try:
            # Preprocess
            processed_data = self.preprocess_input(data)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(processed_data)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probability = torch.sigmoid(output).item()
                prediction = 1 if probability >= 0.5 else 0
            
            # Prepare result
            result = {
                'prediction': prediction,
                'prediction_label': 'Fully Paid' if prediction == 1 else 'Charged Off',
                'confidence': max(probability, 1 - probability),
                'risk_assessment': self._get_risk_assessment(probability)
            }
            
            if return_proba:
                result['probability_fully_paid'] = probability
                result['probability_charged_off'] = 1 - probability
            
            return result
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            raise
    
    def predict_batch(self, data):
        """
        Make predictions for multiple loan applications
        
        Args:
            data (pd.DataFrame): Batch of loan application data
            
        Returns:
            pd.DataFrame: Predictions with probabilities
        """
        try:
            # Preprocess
            processed_data = self.preprocess_input(data)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(processed_data)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.sigmoid(outputs).numpy().flatten()
                predictions = (probabilities >= 0.5).astype(int)
            
            # Create results DataFrame
            results = data.copy()
            results['prediction'] = predictions
            results['prediction_label'] = ['Fully Paid' if pred == 1 else 'Charged Off' 
                                         for pred in predictions]
            results['probability_fully_paid'] = probabilities
            results['probability_charged_off'] = 1 - probabilities
            results['confidence'] = np.maximum(probabilities, 1 - probabilities)
            results['risk_assessment'] = [self._get_risk_assessment(prob) 
                                        for prob in probabilities]
            
            return results
            
        except Exception as e:
            print(f"❌ Error making batch predictions: {str(e)}")
            raise
    
    def _get_risk_assessment(self, probability):
        """
        Get risk assessment based on probability
        
        Args:
            probability (float): Probability of loan being fully paid
            
        Returns:
            str: Risk assessment category
        """
        if probability >= 0.8:
            return "Low Risk"
        elif probability >= 0.6:
            return "Medium-Low Risk"
        elif probability >= 0.4:
            return "Medium-High Risk"
        else:
            return "High Risk"
    
    def get_feature_info(self):
        """Get information about required features"""
        feature_descriptions = {
            'dti': 'Debt-to-income ratio (%)',
            'credit_history_length': 'Credit history length (years)',
            'debt_to_credit_ratio': 'Debt to available credit ratio',
            'revol_bal': 'Total revolving credit balance ($)',
            'installment': 'Monthly loan installment ($)',
            'revol_util': 'Revolving credit utilization (%)',
            'int_rate': 'Loan interest rate (%)',
            'annual_inc': 'Annual income ($)',
            'total_credit_lines': 'Total number of credit lines'
        }
        
        return feature_descriptions


def interactive_prediction(predictor):
    """Interactive single prediction mode"""
    print("\n🎯 Interactive Loan Prediction")
    print("=" * 50)
    print("Enter the following information for the loan application:")
    print()
    
    # Get feature info
    feature_info = predictor.get_feature_info()
    
    # Collect input
    data = {}
    for feature, description in feature_info.items():
        while True:
            try:
                value = float(input(f"{description}: "))
                data[feature] = value
                break
            except ValueError:
                print("Please enter a valid number.")
    
    # Make prediction
    print("\n🔄 Making prediction...")
    result = predictor.predict_single(data)
    
    # Display results
    print("\n📊 Prediction Results")
    print("=" * 30)
    print(f"🎯 Prediction: {result['prediction_label']}")
    print(f"📈 Confidence: {result['confidence']:.2%}")
    print(f"⚠️  Risk Assessment: {result['risk_assessment']}")
    print(f"✅ Probability Fully Paid: {result['probability_fully_paid']:.2%}")
    print(f"❌ Probability Charged Off: {result['probability_charged_off']:.2%}")


def batch_prediction(predictor, input_file, output_file):
    """Batch prediction mode"""
    try:
        print(f"📂 Loading data from {input_file}...")
        data = pd.read_csv(input_file)
        
        print(f"📊 Processing {len(data)} loan applications...")
        results = predictor.predict_batch(data)
        
        print(f"💾 Saving results to {output_file}...")
        results.to_csv(output_file, index=False)
        
        # Print summary
        print("\n📈 Batch Prediction Summary")
        print("=" * 40)
        print(f"Total Applications: {len(results)}")
        print(f"Predicted Fully Paid: {(results['prediction'] == 1).sum()}")
        print(f"Predicted Charged Off: {(results['prediction'] == 0).sum()}")
        print(f"Average Confidence: {results['confidence'].mean():.2%}")
        
        # Risk distribution
        risk_dist = results['risk_assessment'].value_counts()
        print("\n🎯 Risk Distribution:")
        for risk, count in risk_dist.items():
            print(f"  {risk}: {count} ({count/len(results):.1%})")
        
        print(f"\n✅ Results saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Error in batch prediction: {str(e)}")
        raise


def sample_prediction(predictor):
    """Run prediction with sample data"""
    print("\n🧪 Sample Prediction")
    print("=" * 30)
    
    # Sample data - representing a typical loan application
    sample_data = {
        'dti': 15.5,  # Debt-to-income ratio
        'credit_history_length': 8.2,  # Credit history in years
        'debt_to_credit_ratio': 0.35,  # Debt to credit ratio
        'revol_bal': 8500.0,  # Revolving balance
        'installment': 450.0,  # Monthly installment
        'revol_util': 42.5,  # Credit utilization
        'int_rate': 12.8,  # Interest rate
        'annual_inc': 65000.0,  # Annual income
        'total_credit_lines': 12  # Total credit lines
    }
    
    print("📋 Sample loan application data:")
    for feature, value in sample_data.items():
        description = predictor.get_feature_info()[feature]
        print(f"  {description}: {value}")
    
    # Make prediction
    result = predictor.predict_single(sample_data)
    
    # Display results
    print("\n📊 Prediction Results")
    print("=" * 30)
    print(f"🎯 Prediction: {result['prediction_label']}")
    print(f"📈 Confidence: {result['confidence']:.2%}")
    print(f"⚠️  Risk Assessment: {result['risk_assessment']}")
    print(f"✅ Probability Fully Paid: {result['probability_fully_paid']:.2%}")
    print(f"❌ Probability Charged Off: {result['probability_charged_off']:.2%}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Loan Prediction Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python inference.py --single                    # Interactive single prediction
    python inference.py --batch input.csv output.csv  # Batch prediction
    python inference.py --sample                     # Run with sample data
        """
    )
    
    parser.add_argument('--single', action='store_true', 
                       help='Interactive single prediction mode')
    parser.add_argument('--batch', nargs=2, metavar=('INPUT', 'OUTPUT'),
                       help='Batch prediction mode: INPUT_FILE OUTPUT_FILE')
    parser.add_argument('--sample', action='store_true',
                       help='Run prediction with sample data')
    parser.add_argument('--model-path', default='bin/best_checkpoint.pth',
                       help='Path to model checkpoint (default: bin/best_checkpoint.pth)')
    
    args = parser.parse_args()
    
    # Check if no arguments provided
    if not any([args.single, args.batch, args.sample]):
        parser.print_help()
        return
    
    try:
        # Initialize predictor
        print("🚀 Initializing Loan Predictor...")
        predictor = LoanPredictor(model_path=args.model_path)
        
        # Execute based on mode
        if args.single:
            interactive_prediction(predictor)
        elif args.batch:
            batch_prediction(predictor, args.batch[0], args.batch[1])
        elif args.sample:
            sample_prediction(predictor)
            
    except Exception as e:
        print(f"💥 Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
