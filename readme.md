# Credit Card Fraud Detection with Cost-Sensitive Machine Learning

End-to-end machine learning project for highly imbalanced credit card fraud detection using exploratory data analysis, feature engineering, resampling strategies, ensemble learning, and business-cost optimization.

**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Project Overview

This project builds a complete fraud detection workflow from data exploration to final model comparison. The dataset is extremely imbalanced (about 0.17% fraud), so the pipeline is designed to prevent leakage and prioritize minority-class performance.

**Main objectives:**
- Detect fraudulent transactions with high recall and strong precision-recall performance
- Compare classical, ensemble, and cost-sensitive learning approaches
- Optimize prediction thresholds using business cost instead of default 0.5

## Highlights

- **EDA:** Class imbalance analysis, transaction amount/time distributions, correlation heatmaps, boxplots, PCA/t-SNE/Truncated SVD clustering
- **Feature Engineering:** Pearson correlation, mutual information selection
- **Imbalance Handling:** Random undersampling, SMOTE, cost-sensitive weighting
- **Models:** Classical ML, ensemble methods (Balanced Random Forest, XGBoost, LightGBM)
- **Evaluation:** PR-AUC, recall, F1, ROC-AUC, business cost optimization

## Project Structure

```
├── credit_fraud_complete.ipynb
├── creditcard.csv
├── requirements.txt
├── images/
└── resources/
```

## Tech Stack

Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn, XGBoost, LightGBM

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install imbalanced-ensemble  # Optional
```

## How to Run

1. Open `credit_fraud_complete.ipynb`
2. Run cells sequentially
3. Review EDA, feature engineering, model comparisons, and threshold optimization

## Key Findings

- PR-AUC is most informative for imbalanced fraud detection
- Fold-level resampling prevents data leakage
- Ensemble and cost-sensitive methods outperform baselines
- Threshold optimization provides substantial business value

## Author

Shalindra Perera

## License

Educational and research use only.


## Questions

### Are there any Standard Benchmarks for imbalanced classification problems?
Yes, there are several standard benchmarks and datasets commonly used for evaluating imbalanced classification algorithms. Some of the most popular ones include:
1. CLIMB Dataset: A collection of datasets specifically designed for imbalanced classification tasks, covering various domains such as medical diagnosis, fraud detection, and more.
2. KEEL Dataset Repository: A repository of datasets for imbalanced classification, regression, and clustering tasks, often used for benchmarking algorithms.   
3. UCI Machine Learning Repository: Contains several datasets that are imbalanced, such as the Credit Card Fraud Detection dataset, which is widely used for benchmarking imbalanced classification algorithms.
4. MNIST Imbalanced: A modified version of the MNIST dataset where certain classes are underrepresented, often used for benchmarking imbalanced classification algorithms in image recognition tasks.

### What does OOF mean?

**OOF** = Out-Of-Fold

When you train a model using cross-validation (splitting data into multiple folds), the OOF predictions are the predictions made on each fold while that fold was held out (not used for training).

Think of it like this:

1. Split data into 5 parts
2. Train on parts 1,2,3,4 → predict on part 5 → those are OOF predictions
3. Train on parts 1,2,3,5 → predict on part 4 → OOF again
4. ...and so on

At the end, every sample gets one OOF prediction — made by a model that never saw that sample during training. This gives you an honest, unbiased estimate of how your model performs.