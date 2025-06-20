# 🏦 Loan Prediction System

A comprehensive machine learning system for predicting loan approval decisions using deep neural networks. This project implements an end-to-end ML pipeline with exploratory data analysis, feature engineering, model training, and deployment capabilities.

## 📊 Project Overview

This project uses the LendingClub dataset to build a robust loan prediction model that helps financial institutions make data-driven lending decisions. The system achieves **70.1% accuracy** with **86.4% precision** using a deep neural network architecture.

### Key Features

- **Advanced EDA**: Comprehensive exploratory data analysis with feature engineering
- **Deep Learning Model**: Multi-layer neural network with dropout regularization
- **Production Ready**: Streamlit web application for real-time predictions
- **Robust Pipeline**: End-to-end ML pipeline with data preprocessing and model training
- **Performance Monitoring**: Detailed metrics and visualization tools

## 🎯 Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 70.1% |
| Precision | 86.4% |
| Recall | 74.5% |
| F1-Score | 80.0% |
| AUC-ROC | 69.0% |

## 🏗️ Architecture

### Model Architecture
- **Input Layer**: 9 features (after feature selection)
- **Hidden Layers**: 
  - Layer 1: 128 neurons (ReLU, Dropout 0.3)
  - Layer 2: 64 neurons (ReLU, Dropout 0.3)
  - Layer 3: 32 neurons (ReLU, Dropout 0.2)
  - Layer 4: 16 neurons (ReLU, Dropout 0.1)
- **Output Layer**: 1 neuron (Sigmoid activation)

### Project Structure

```
loan_prediction/
├── README.md                 # Main project documentation
├── requirements.txt          # Python dependencies
├── src/                      # Source code
│   ├── model.py             # Neural network architecture
│   ├── train.py             # Training pipeline
│   └── inference.py         # Inference and prediction
├── scripts/                  # Utility scripts
│   └── app.py               # Streamlit web application
├── notebooks/               # Jupyter notebooks
│   └── EDA.ipynb           # Exploratory data analysis
├── docs/                    # Documentation
│   ├── EDA_README.md       # EDA decisions and methodology
│   └── MODEL_ARCHITECTURE.md # Model design details
├── data/                    # Data files
│   ├── lending_club_loan_two.csv
│   ├── lending_club_info.csv
│   └── processed/          # Processed data files
├── bin/                     # Model checkpoints
│   └── best_checkpoint.pth
└── __pycache__/            # Python cache files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Streamlit 1.28+

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd loan_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**
   ```bash
   streamlit run scripts/app.py
   ```

### Training the Model

```bash
python src/train.py
```

### Making Predictions

```bash
# Interactive single prediction
python src/inference.py --single

# Batch prediction
python src/inference.py --batch input.csv output.csv

# Sample prediction
python src/inference.py --sample
```

## 📋 Usage Examples

### Web Application
Launch the Streamlit app for an interactive loan prediction interface:
```bash
streamlit run scripts/app.py
```

### Command Line Inference
```bash
# Single prediction with interactive input
python src/inference.py --single

# Batch processing
python src/inference.py --batch data/test_file.csv results/predictions.csv
```

### Training Custom Model
```bash
python src/train.py --epochs 200 --batch_size 1536 --learning_rate 0.012
```

## 📈 Data & Features

### Dataset
- **Source**: LendingClub loan data
- **Size**: ~400,000 loan records
- **Features**: 23 original features reduced to 9 after feature selection

### Selected Features
1. **loan_amnt**: Loan amount requested
2. **int_rate**: Interest rate on the loan
3. **installment**: Monthly payment amount
4. **grade**: LC assigned loan grade
5. **emp_length**: Employment length in years
6. **annual_inc**: Annual income
7. **dti**: Debt-to-income ratio
8. **open_acc**: Number of open credit accounts
9. **pub_rec**: Number of derogatory public records

## 📚 Documentation

- **[EDA Analysis & Decisions](docs/EDA_README.md)** - Detailed explanation of exploratory data analysis and feature engineering decisions
- **[Model Architecture](docs/MODEL_ARCHITECTURE.md)** - Deep dive into neural network design and training methodology

## 🔧 Configuration

### Training Configuration
```json
{
  "learning_rate": 0.012,
  "batch_size": 1536,
  "num_epochs": 200,
  "early_stopping_patience": 30,
  "weight_decay": 0.0001,
  "validation_split": 0.2
}
```

## 📊 Model Performance

### Training History
- **Best Epoch**: Achieved at epoch 112
- **Training Loss**: Converged to ~0.32
- **Validation Loss**: Stabilized at ~0.34
- **Early Stopping**: Activated after 30 epochs without improvement

### Class Distribution
- **Default Rate**: ~22% (imbalanced dataset)
- **Handling**: Weighted loss function and class balancing techniques

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LendingClub for providing the dataset
- PyTorch team for the deep learning framework
- Streamlit for the web application framework

## 📞 Contact

For questions or support, please open an issue in the repository.

---

**Note**: This model is for educational and research purposes. Always consult with financial experts before making actual lending decisions.
