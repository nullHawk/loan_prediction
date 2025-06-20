"""
Unit tests for model functionality
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.model import LoanPredictionDeepANN

class TestLoanPredictionModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model = LoanPredictionDeepANN(input_size=9)
        self.sample_input = torch.randn(10, 9)  # Batch of 10 samples
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, LoanPredictionDeepANN)
        self.assertEqual(self.model.fc1.in_features, 9)
        self.assertEqual(self.model.fc5.out_features, 1)
    
    def test_forward_pass(self):
        """Test forward pass"""
        output = self.model(self.sample_input)
        
        # Check output shape
        self.assertEqual(output.shape, (10, 1))
        
        # Check output range (should be between 0 and 1 due to sigmoid)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_model_parameters(self):
        """Test model has parameters"""
        params = list(self.model.parameters())
        self.assertTrue(len(params) > 0)
        
        # Check parameter shapes
        self.assertEqual(params[0].shape, (128, 9))  # First layer weights
        self.assertEqual(params[1].shape, (128,))    # First layer bias
    
    def test_training_mode(self):
        """Test training and eval modes"""
        self.model.train()
        self.assertTrue(self.model.training)
        
        self.model.eval()
        self.assertFalse(self.model.training)

if __name__ == '__main__':
    unittest.main()
