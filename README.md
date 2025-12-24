# Fake Job Posting Detection System

# Overview
Online job portals are increasingly targeted by fraudulent job postings, leading to financial and personal data loss for applicants.  
This project builds a **machine learning–based detection system** to classify job postings as real or fake using textual and metadata features.

The goal is not just high accuracy, but **robust detection of high-risk fraudulent postings**, prioritizing recall and real-world applicability.

# Dataset
- Source: Publicly available job posting dataset
- Data characteristics:
  - Highly imbalanced classes (fake vs real)
  - Noisy text fields and missing metadata
- Preprocessing included handling missing values, text normalization, and label consistency checks.

# Approach
1. **Data Cleaning & Preprocessing**
   - Text normalization (lowercasing, punctuation removal)
   - Tokenization and stopword removal
   - Handling class imbalance

2. **Feature Engineering**
   - TF-IDF vectorization for textual fields
   - Combination of textual and structured features

3. **Modeling**
   - Logistic Regression
   - Naive Bayes
   - Random Forest
   - Comparative evaluation across models

4. **Evaluation Strategy**
   - Precision, Recall, F1-score
   - ROC-AUC
   - Emphasis on minimizing false negatives (undetected fake jobs)

# Results & Observations
- Linear models performed well on high-dimensional sparse text features.
- Random Forest showed better recall in certain configurations.
- Trade-off observed between precision and recall depending on model choice.
- Highlighted the importance of feature selection and data quality.

# Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- NLP preprocessing tools

# Future Work
- Incorporate contextual embeddings (e.g., transformer-based models)
- Apply cost-sensitive learning for imbalanced classification
- Extend system for real-time job posting monitoring
