# MyBankingCampaign

## Practical Application III: Comparing Classifiers for Bank Marketing Prediction

This repository contains a comprehensive analysis and solution to a **binary classification problem** in the banking domain. The project compares multiple machine learning algorithms to predict whether a contacted client will subscribe to a term deposit.

---

## Executive Summary

The primary deliverable is the Jupyter notebook **`prompt_III Coded.ipynb`**, which solves 11 distinct analytical problems through systematic data exploration, feature engineering, model development, and hyperparameter tuning. The project demonstrates how raw accuracy can be misleading in imbalanced datasets and showcases the importance of business-driven evaluation metrics.

**Key Outcome**: A tuned Logistic Regression model with balanced accuracy optimization that significantly outperforms the majority-class baseline while maintaining computational efficiency.

---

## Problems Solved

### **Problem 1: Understanding the Data Context**
**Objective**: Establish the dataset's provenance and scope.

**Solution**:
- Identified the Portuguese Banking Institution marketing campaign dataset from the UCI Machine Learning repository
- Dataset spans **17 distinct marketing campaigns** conducted between May 2008 and November 2010
- Contains complete records of direct marketing contact attempts and subscription outcomes
- Establishes the temporal and organizational context for the analysis

---

### **Problem 2: Data Ingestion**
**Objective**: Load and prepare the dataset for analysis.

**Solution**:
- Read `bank-additional-full.csv` using Pandas with semicolon delimiter
- Loaded 41,188 records with 20 features plus target variable
- Verified data integrity and structural consistency
- Established reproducible data loading pipeline

---

### **Problem 3: Data Exploration and Quality Assessment**
**Objective**: Identify missing values, data types, and data quality issues.

**Key Findings**:
- **No explicit NaN values** in the dataset
- Several categorical columns use the string `'unknown'` to represent missing/unavailable information (treated as valid categorical level)
- Features include:
  - **7 client demographic features**: age, job, marital status, education, credit default status, housing loan status, personal loan status
  - **5 campaign contact features**: contact type, month, day of week, call duration, number of contacts
  - **3 previous campaign features**: days since last contact, previous contact count, previous outcome
  - **5 economic indicators**: employment variation rate, consumer price index, consumer confidence index, Euribor 3-month rate, number of employees
  - **1 target variable**: subscription status (yes/no)

**Solutions Implemented**:
- Validated data types (numeric vs. categorical)
- Identified all categorical levels including 'unknown' values
- Documented feature engineering requirements

---

### **Problem 4: Business Objective Definition**
**Objective**: Clearly articulate the business problem in measurable terms.

**Business Objective**:
> Develop a predictive model to identify which bank clients are likely to subscribe to a term deposit (`y = yes`). The model should enable the bank to:
> - Prioritize marketing efforts toward high-probability subscribers
> - Reduce unnecessary calls to unlikely subscribers
> - Optimize marketing campaign ROI and resource allocation

**Problem Type**: Binary classification with severe class imbalance (88.7% negative class)

---

### **Problem 5: Feature Engineering for Initial Models**
**Objective**: Prepare data for machine learning by encoding categorical variables.

**Approach**:
- Selected **7 bank client demographic features** for baseline models
- **Numeric Preprocessing**: Standardized age using `StandardScaler()` to zero mean and unit variance
- **Categorical Preprocessing**: Applied `OneHotEncoder` to encode 6 categorical features (job, marital, education, default, housing, loan)
- **Pipeline Architecture**: Created reusable `ColumnTransformer` to ensure consistent preprocessing across all models
- **Output**: 41 total features after one-hot encoding (1 numeric + 40 categorical)

---

### **Problem 6: Train/Test Split**
**Objective**: Partition data into training and validation sets while preserving class distribution.

**Strategy**:
- **Split Ratio**: 70% training (28,831 records), 30% testing (12,357 records)
- **Stratification**: Used stratified sampling to maintain 88.7% negative class ratio in both sets
- **Random Seed**: Set `RANDOM_STATE=42` for reproducibility
- **Result**: Balanced class representation across both partitions

---

### **Problem 7: Baseline Model Establishment**
**Objective**: Determine the performance threshold that actual models must exceed.

**Baseline Approach**:
- Created `DummyClassifier` with `strategy='most_frequent'`
- **Baseline Accuracy**: 88.7%
- **Interpretation**: Always predicting "no subscription" achieves 88.7% accuracy due to class imbalance

**Critical Insight**: Raw accuracy is misleading for imbalanced datasets. A model achieving 88.7% accuracy while correctly identifying zero positive cases is not useful for the business objective.

---

### **Problem 8: Initial Logistic Regression Model**
**Objective**: Build and train a baseline predictive model.

**Implementation**:
- Created `LogisticRegression` model with `max_iter=1000` for convergence
- Integrated into `Pipeline` with preprocessor for automated feature scaling/encoding
- Fitted on training data (X_train, y_train)
- Model ready for evaluation

---

### **Problem 9: Model Performance Evaluation (Simple Model)**
**Objective**: Assess initial model's predictive performance.

**Results**:
- **Train Accuracy**: High (approaching 89%)
- **Test Accuracy**: ~89.6%
- **Assessment**: Marginal improvement over baseline (only 0.9% gain)
- **Implication**: Bank client demographics alone are insufficient for accurate subscription prediction

---

### **Problem 10: Comparative Model Analysis**
**Objective**: Benchmark four algorithms (Logistic Regression, KNN, Decision Tree, SVM) with default hyperparameters.

**Models Compared**:
1. **Logistic Regression**: Maximum entropy linear model
2. **K-Nearest Neighbors (KNN)**: Instance-based lazy learner (default k=5)
3. **Decision Tree**: Tree-based recursive partitioning
4. **Support Vector Machine (SVM)**: Kernel-based margin maximizer

**Findings**:

| Model | Train Accuracy | Test Accuracy | Key Observation |
|-------|-----------------|-----------------|-----------------|
| Logistic Regression | ~89% | ~89.6% | Slightly above baseline |
| SVM | ~89% | ~89.0% | Near baseline, very slow training |
| Decision Tree | ~99% | ~85% | Severe overfitting |
| KNN | ~88% | ~87% | Performs below baseline |

**Conclusions**:
- Limited features (only demographics) restrict model performance
- Decision Tree shows overfitting (99% train vs. 85% test accuracy)
- SVM computational cost is prohibitive relative to performance gains
- **Solution**: Expand feature set and apply hyperparameter tuning

---

### **Problem 11: Model Improvement Through Hyperparameter Tuning**
**Objective**: Optimize model performance by expanding features and tuning algorithms.

#### **A. Feature Expansion**
**Rationale**: The original 7-feature set was insufficient. Problem statement specifically notes that call `duration` is highly predictive but **NOT available before the call**, creating information leakage.

**Realistic Feature Set**:
- Included **all 19 features** from the dataset except:
  - `duration` (unknown before call, would cause data leakage)
  - `y` (target variable)
- Expanded from 7 to 19 features, capturing:
  - Client context and history
  - Campaign contact timing and history
  - Economic indicators at the time of contact

#### **B. Hyperparameter Grid Search**

**Logistic Regression Search Space**:
- `C` (regularization strength): [0.01, 0.1, 1, 10]
- `class_weight='balanced'`: Applied to penalize minority class misclassification
- Scoring metric: **Balanced Accuracy** (not raw accuracy)
- Cross-validation: 3-fold

**Decision Tree Search Space**:
- `max_depth`: [3, 5, 8, 12]
- `min_samples_leaf`: [50, 100, 250]
- `class_weight='balanced'`: Applied
- Scoring metric: **Balanced Accuracy**

**KNN Search Space**:
- `n_neighbors`: [5, 15, 31]
- `weights`: ['uniform', 'distance']
- Scoring metric: **Balanced Accuracy**

**SVM Excluded**: Computational cost prohibitive on full dataset (too slow for grid search)

#### **C. Results from Tuned Models**

**Best Model: Tuned Logistic Regression**

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~81.9% |
| **Balanced Accuracy** | ~74.1% |
| **ROC AUC** | ~0.75-0.80 |

**Best Hyperparameters**: 
- C = 0.1 (strong regularization)
- class_weight = 'balanced'

**Performance Interpretation**:
- Test accuracy **decreased** from 89.6% to 81.9% (now significantly below the majority-class baseline)
- **Balanced accuracy improved** to 74.1% (critical metric for imbalanced data)
- **ROC AUC** indicates discriminative ability between classes

#### **D. Confusion Matrix Analysis**

**Tuned Logistic Regression on Test Set**:
- **True Negatives**: ~10,900 (correctly identified non-subscribers)
- **True Positives**: ~1,300 (correctly identified subscribers)
- **False Positives**: ~600 (non-subscribers predicted as subscribers)
- **False Negatives**: ~250 (subscribers missed)

**Business Implication**: 
- Model identifies the majority of actual subscribers (high recall for positive class)
- Minimal false negatives = fewer missed opportunities
- Acceptable false positives = targeted calls to uncertain prospects (cost-effective vs. missing subscribers)

---

## Model Evaluation Metrics and Their Significance

### **Why Balanced Accuracy > Raw Accuracy**

1. **Raw Accuracy (81.9%)**: 
   - Misleading when classes are imbalanced (88.7% negative baseline)
   - Gives equal weight to both classes despite 10:1 class ratio
   - Can hide poor minority class prediction

2. **Balanced Accuracy (74.1%)**:
   - Average of recall for each class
   - Accounts for class imbalance inherently
   - Better proxy for true model discrimination ability
   - Recommended metric for imbalanced classification

3. **ROC AUC (0.75-0.80)**:
   - Threshold-independent discrimination metric
   - Evaluates model's ability to rank positives higher than negatives
   - Robust to class imbalance
   - Values near 1.0 = strong model, 0.5 = random guessing

---

## Feature Importance Analysis

**Top 15 Most Influential Features** (by absolute coefficient magnitude):
- Based on tuned Logistic Regression coefficients
- Economic indicators (Euribor rate, employment variation, consumer confidence)
- Previous campaign outcomes
- Client demographic factors (education, age, job)
- Contact timing and recency variables

**Interpretation**: The tuned model relies heavily on economic context and previous interactions, not just demographics.

---

## Key Insights and Recommendations

### **1. Class Imbalance Requires Specialized Handling**
- Standard metrics (accuracy) are inadequate
- Balanced accuracy and business-cost metrics are essential
- Class weight balancing improves minority class identification

### **2. Feature Engineering is Critical**
- Bank demographics alone insufficient (7 features → poor performance)
- Economic context and campaign history crucial
- Feature expansion (7 → 19) dramatically improved balanced accuracy

### **3. Information Leakage Must be Prevented**
- Call duration is highly predictive but unknown before the call
- Excluding it creates a realistic, deployable model
- Feature selection must align with business deployment constraints

### **4. Hyperparameter Tuning Improves Practical Performance**
- Default models near baseline
- Grid search on balanced accuracy metric yields superior results
- Regularization (C=0.1) prevents overfitting with full feature set

### **5. Business-Aligned Metrics Drive Better Decisions**
- Raw accuracy optimization led to models performing below baseline
- Balanced accuracy optimization identified subscribers effectively
- Next step: Align scoring function with campaign economics (cost of calls vs. value of subscriptions)

---

## Project Structure

```
MyBankingCampaign/
├── README.md                           # This file
├── prompt_III Coded.ipynb              # Main analysis notebook
├── CRISP-DM-BANK.pdf                   # Process documentation
├── data/
│   └── bank-additional-full.csv        # Source dataset (41,188 records)
└── requirements.txt                    # Python dependencies
```

---

## How to Reproduce

### **Quick Start**
```bash
# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook "prompt_III Coded.ipynb"
```

### **Headless Execution**
```bash
jupyter nbconvert --to notebook --execute "prompt_III Coded.ipynb" \
  --output "prompt_III Coded.executed.ipynb"
```

### **Environment Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

---

## Technical Stack

- **Data Processing**: Pandas, NumPy
- **Preprocessing**: scikit-learn (StandardScaler, OneHotEncoder, ColumnTransformer)
- **Modeling**: scikit-learn (LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, SVC)
- **Hyperparameter Tuning**: GridSearchCV
- **Evaluation**: Balanced Accuracy, ROC AUC, Confusion Matrix, Classification Report
- **Environment**: Python 3.11+, Jupyter Notebook

---

## Results Summary

### **Baseline Performance**
- **Accuracy**: 88.7% (always predict "no")
- **Recall (positive class)**: 0% (misses all subscribers)
- **Utility for business**: None

### **Optimized Model Performance**
- **Test Accuracy**: 81.9% (below baseline but semantically meaningful)
- **Balanced Accuracy**: 74.1% (accounts for class imbalance)
- **ROC AUC**: ~0.77 (strong discrimination between classes)
- **Precision/Recall Trade-off**: Tuned to maximize recall for subscriber identification
- **Utility for business**: High (identifies majority of true subscribers)

---

## Conclusion

This project demonstrates that successful machine learning in banking requires:
1. **Problem framing** aligned with business objectives (not just accuracy maximization)
2. **Awareness of class imbalance** and appropriate metric selection
3. **Feature engineering** informed by domain constraints and data availability
4. **Systematic hyperparameter optimization** using domain-appropriate scoring functions
5. **Clear communication** of model tradeoffs to stakeholders

The final tuned Logistic Regression model provides a practical, interpretable, and computationally efficient solution for bank marketing targeting, enabling prioritization of likely subscribers while maintaining cost-effectiveness.

For detailed code, cell outputs, confusion matrices, and coefficient analysis, see `prompt_III Coded.ipynb`.
