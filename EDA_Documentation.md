# Loan Prediction EDA Documentation

## Executive Summary

This document provides a comprehensive overview of the Exploratory Data Analysis (EDA) and Feature Engineering process performed on the Lending Club loan dataset for training an Artificial Neural Network (ANN) to predict loan repayment outcomes.

**Dataset**: Lending Club Loan Data  
**Original Size**: 396,030 records × 27 features  
**Final Processed Size**: 396,030 records × 9 features  
**Target Variable**: Loan repayment status (binary classification)  
**Date**: June 2025

---

## Table of Contents

1. [Data Overview](#data-overview)
2. [Initial Data Exploration](#initial-data-exploration)
3. [Missing Data Analysis](#missing-data-analysis)
4. [Target Variable Analysis](#target-variable-analysis)
5. [Feature Correlation Analysis](#feature-correlation-analysis)
6. [Categorical Feature Analysis](#categorical-feature-analysis)
7. [Feature Engineering](#feature-engineering)
8. [Feature Selection](#feature-selection)
9. [Data Preprocessing for ANN](#data-preprocessing-for-ann)
10. [Final Dataset Summary](#final-dataset-summary)

---

## 1. Data Overview

### Initial Dataset Structure
- **Shape**: 396,030 rows × 27 columns
- **Target Variable**: `loan_status` (Fully Paid vs Charged Off)
- **Feature Types**: Mix of numerical and categorical variables
- **Domain**: Peer-to-peer lending data from Lending Club

### Key Business Context
The goal is to predict whether a borrower will fully repay their loan or default (charge off). This is a critical business problem for lenders as it directly impacts:
- Risk assessment
- Interest rate pricing
- Portfolio management
- Regulatory compliance

---

## 2. Initial Data Exploration

### Why This Step Was Performed
Understanding the basic structure and characteristics of the dataset is crucial before any analysis. This helps identify:
- Data quality issues
- Feature types and distributions
- Potential preprocessing needs

### Actions Taken
```python
# Basic exploration commands used:
df.shape          # Dataset dimensions
df.info()         # Data types and memory usage
df.describe()     # Statistical summary for numerical features
df.columns        # Feature names
```

### Key Findings
- 396,030 loan records spanning multiple years
- Mix of numerical (interest rates, amounts, ratios) and categorical (grades, purposes) features
- Presence of date features requiring special handling
- Some features with high cardinality (e.g., employment titles)

---

## 3. Missing Data Analysis

### Why This Step Was Critical
Missing data can significantly impact model performance and introduce bias. For neural networks, complete data is especially important for stable training.

### Methodology
1. **Quantified missing values** for each feature
2. **Visualized missing patterns** using heatmap
3. **Applied strategic removal and imputation**

### Actions Taken
```python
# Missing data analysis
df.isnull().sum().sort_values(ascending=False)
sns.heatmap(df.isnull(), cbar=False)  # Visual pattern analysis
```

### Decisions Made
1. **Removed high-missing features**: 
   - `mort_acc` (mortgage accounts)
   - `emp_title` (employment titles - too many unique values)
   - `emp_length` (employment length - high missingness)
   - `title` (loan titles - redundant with purpose)

2. **Imputation strategy**:
   - **Numerical features**: Median imputation (robust to outliers)
   - **Categorical features**: Mode imputation (most frequent category)

### Rationale
- Features with >50% missing data were dropped to avoid introducing too much imputed noise
- Median imputation chosen over mean for numerical features due to potential skewness in financial data
- Mode imputation maintains the natural distribution of categorical variables

---

## 4. Target Variable Analysis

### Why This Analysis Was Essential
Understanding target distribution is crucial for:
- Identifying class imbalance
- Choosing appropriate evaluation metrics
- Determining if sampling techniques are needed

### Findings
- **Fully Paid**: 318,357 loans (80.4%)
- **Charged Off**: 77,673 loans (19.6%)
- **Class Ratio**: ~4:1 (moderate imbalance)

### Target Engineering Decision
Created binary target variable `loan_repaid`:
- **1**: Fully Paid (positive outcome)
- **0**: Charged Off (negative outcome)

### Impact on Modeling
The 80/20 split represents a moderate class imbalance that's manageable for neural networks without requiring aggressive resampling techniques.

---

## 5. Feature Correlation Analysis

### Purpose
Identify relationships between numerical features and the target variable to:
- Understand predictive power of individual features
- Detect potential multicollinearity issues
- Guide feature selection priorities

### Methodology
```python
# Correlation analysis with target
correlation_with_target = df[numerical_features + ['loan_repaid']].corr()['loan_repaid']
```

### Key Discoveries
**Top Predictive Features** (by correlation magnitude):
1. `revol_util` (-0.082): Higher revolving credit utilization = higher default risk
2. `dti` (-0.062): Higher debt-to-income ratio = higher default risk
3. `loan_amnt` (-0.060): Larger loans = higher default risk
4. `annual_inc` (+0.053): Higher income = lower default risk

### Business Insights
- **Credit utilization** emerged as the strongest single predictor
- **Debt ratios** consistently showed negative correlation with repayment
- **Income level** showed positive correlation with successful repayment
- Correlations were relatively weak, suggesting need for feature engineering

---

## 6. Categorical Feature Analysis

### Objective
Understand how categorical variables relate to loan outcomes and identify high-impact categories.

### Features Analyzed
- `grade`: Lending Club's risk assessment (A-G)
- `home_ownership`: Housing status
- `verification_status`: Income verification level
- `purpose`: Loan purpose
- `initial_list_status`: Initial listing status
- `application_type`: Individual vs joint application

### Key Findings

#### Grade Analysis
- **Grade A**: ~95% repayment rate (highest quality)
- **Grade G**: ~52% repayment rate (highest risk)
- Clear monotonic relationship between grade and repayment rate

#### Home Ownership
- **Any/Other**: Highest repayment rates (~100%)
- **Rent**: Lowest repayment rates (~78%)
- **Own/Mortgage**: Middle performance (~80-83%)

#### Purpose Analysis
- **Wedding**: Highest repayment rate (~88%)
- **Small Business**: Lowest repayment rate (~71%)
- **Debt Consolidation**: Most common purpose with ~80% repayment

### Business Implications
- Lending Club's internal grading system is highly predictive
- Housing stability correlates with loan performance
- Loan purpose provides significant risk differentiation

---

## 7. Feature Engineering

### Strategic Approach
Created new features to capture complex relationships and domain knowledge that raw features might miss.

### New Features Created

#### Date-Based Features
```python
df['credit_history_length'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 365.25
df['issue_year'] = df['issue_d'].dt.year
df['issue_month'] = df['issue_d'].dt.month
```
**Rationale**: Credit history length is a key risk factor in traditional credit scoring.

#### Financial Ratio Features
```python
df['debt_to_credit_ratio'] = df['revol_bal'] / (df['revol_bal'] + df['annual_inc'] + 1)
df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)
df['installment_to_income'] = df['installment'] / (df['annual_inc'] / 12 + 1)
```
**Rationale**: Ratios normalize absolute amounts and capture relative financial stress.

#### Credit Utilization
```python
df['credit_utilization_ratio'] = df['revol_util'] / 100
```
**Rationale**: Convert percentage to ratio for consistent scaling.

#### Risk Encoding
```python
grade_mapping = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
df['grade_numeric'] = df['grade'].map(grade_mapping)
```
**Rationale**: Convert ordinal risk grades to numerical values preserving order.

#### Aggregate Features
```python
df['total_credit_lines'] = df['open_acc'] + df['total_acc']
```
**Rationale**: Total credit experience indicator.

### Feature Engineering Validation
- Checked for infinite and NaN values in all new features
- Verified logical ranges and distributions
- Confirmed business logic alignment

---

## 8. Feature Selection

### Multi-Stage Selection Process

#### Stage 1: Categorical Encoding
Applied Label Encoding to categorical variables for compatibility with numerical analysis methods.

#### Stage 2: Random Forest Feature Importance
```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_temp, y_temp)
feature_importance = rf.feature_importances_
```

**Why Random Forest for Feature Selection:**
- Handles mixed data types well
- Captures non-linear relationships
- Provides relative importance scores
- Less prone to overfitting than single trees

#### Stage 3: Top Features Identification
Selected top 15 features based on importance scores:

1. **dti** (0.067): Debt-to-income ratio
2. **loan_to_income_ratio** (0.061): Loan amount relative to income
3. **credit_history_length** (0.061): Years of credit history
4. **installment_to_income** (0.060): Monthly payment burden
5. **debt_to_credit_ratio** (0.058): Debt utilization measure
6. **revol_bal** (0.057): Revolving credit balance
7. **installment** (0.054): Monthly payment amount
8. **revol_util** (0.053): Revolving credit utilization
9. **int_rate** (0.053): Interest rate
10. **credit_utilization_ratio** (0.053): Utilization as ratio
11. **annual_inc** (0.050): Annual income
12. **total_credit_lines** (0.045): Total credit accounts
13. **sub_grade_encoded** (0.045): Detailed risk grade
14. **total_acc** (0.044): Total accounts ever
15. **loan_amnt** (0.043): Loan amount

#### Stage 4: Multicollinearity Removal
Identified and removed highly correlated features (r > 0.8):

**Removed Features and Rationale:**
- `loan_to_income_ratio` (r=0.884 with dti): Keep dti as more standard metric
- `installment_to_income` (r=0.977 with loan_to_income_ratio): Redundant information
- `credit_utilization_ratio` (r=1.000 with revol_util): Perfect correlation
- `sub_grade_encoded` (r=0.974 with int_rate): Interest rate more direct
- `total_acc` (r=0.971 with total_credit_lines): Keep engineered feature
- `loan_amnt` (r=0.954 with installment): Monthly impact more relevant

### Final Feature Set (9 features)
1. **dti**: Debt-to-income ratio
2. **credit_history_length**: Credit history in years
3. **debt_to_credit_ratio**: Debt utilization measure
4. **revol_bal**: Revolving balance amount
5. **installment**: Monthly payment amount
6. **revol_util**: Revolving utilization percentage
7. **int_rate**: Interest rate
8. **annual_inc**: Annual income
9. **total_credit_lines**: Total credit accounts

---

## 9. Data Preprocessing for ANN

### Why These Steps Were Necessary
Neural networks are sensitive to:
- Feature scale differences
- Input distribution characteristics
- Data leakage between train/test sets

### Preprocessing Pipeline

#### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_final
)
```
**Parameters Chosen:**
- **80/20 split**: Standard for large datasets
- **Stratified**: Maintains class balance in both sets
- **Random state**: Ensures reproducibility

#### Feature Scaling
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why StandardScaler:**
- **Neural networks benefit from normalized inputs** (typically mean=0, std=1)
- **Prevents feature dominance** based on scale
- **Improves gradient descent convergence**
- **Fit only on training data** to prevent data leakage

### Data Leakage Prevention
- Scaler fitted only on training data
- All transformations applied consistently to test data
- No future information used in feature creation

---

## 10. Final Dataset Summary

### Dataset Characteristics
- **Training Set**: 316,824 samples (80%)
- **Test Set**: 79,206 samples (20%)
- **Features**: 9 carefully selected numerical features
- **Target Distribution**: Maintained 80.4% Fully Paid, 19.6% Charged Off

### Feature Quality Metrics
- **Maximum correlation between features**: 0.632 (acceptable level)
- **All features scaled**: Mean ≈ 0, Standard deviation ≈ 1
- **No missing values**: Complete dataset ready for training
- **Feature importance range**: 0.043 to 0.067 (balanced contribution)

### Model Readiness Checklist
✅ **No missing values**  
✅ **Appropriate feature scaling**  
✅ **Balanced feature importance**  
✅ **Minimal multicollinearity**  
✅ **Stratified train-test split**  
✅ **Class distribution preserved**  
✅ **No data leakage**  

### Business Value Preserved
The final feature set maintains strong business interpretability:
- **Financial ratios**: dti, debt_to_credit_ratio, revol_util
- **Credit behavior**: credit_history_length, total_credit_lines
- **Loan characteristics**: int_rate, installment
- **Financial capacity**: annual_inc, revol_bal

---

## Methodology Strengths

### 1. Domain-Driven Approach
- Feature engineering based on credit risk principles
- Business logic validation at each step
- Interpretable feature selection

### 2. Statistical Rigor
- Systematic missing data analysis
- Correlation-based multicollinearity detection
- Stratified sampling for train-test split

### 3. Model-Appropriate Preprocessing
- Standardization suitable for neural networks
- Feature selection optimized for predictive power
- Data leakage prevention measures

### 4. Reproducibility
- Fixed random seeds throughout
- Documented preprocessing steps
- Saved preprocessing parameters

---

## Recommendations for ANN Training

### 1. Architecture Suggestions
- **Input layer**: 9 neurons (one per feature)
- **Hidden layers**: Start with 2-3 layers, 16-32 neurons each
- **Output layer**: 1 neuron with sigmoid activation (binary classification)

### 2. Training Considerations
- **Class weights**: Consider using class_weight='balanced' due to 80/20 split
- **Regularization**: Dropout layers (0.2-0.3) to prevent overfitting
- **Early stopping**: Monitor validation loss to prevent overtraining

### 3. Evaluation Metrics
- **Primary**: AUC-ROC (handles class imbalance well)
- **Secondary**: Precision, Recall, F1-score
- **Business**: False positive/negative rates and associated costs

### 4. Potential Enhancements
- **Feature interactions**: Consider polynomial features for top variables
- **Ensemble methods**: Combine ANN with tree-based models
- **Advanced sampling**: SMOTE if class imbalance proves problematic

---

## Conclusion

This EDA process transformed a raw dataset of 396,030 loan records with 27 features into a clean, analysis-ready dataset with 9 highly predictive features. The methodology emphasized:

- **Data quality** through systematic missing value handling
- **Feature relevance** through importance-based selection
- **Model compatibility** through appropriate preprocessing
- **Business alignment** through domain-knowledge integration

The resulting dataset is optimally prepared for neural network training while maintaining strong business interpretability and statistical validity.
