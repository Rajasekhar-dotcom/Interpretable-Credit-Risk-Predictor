# 🏦 Interpretable Credit Risk Predictor (XAI)

## Overview
A major challenge in deploying Machine Learning models in the financial sector is the "black box" problem—models provide predictions without context. This project is an end-to-end Explainable AI (XAI) pipeline designed not just to predict credit card defaults, but to provide mathematically rigorous explanations for every single decision.

Built using Python, Pandas, and Scikit-Learn, the pipeline handles heavy class imbalances using **SMOTE** before training a high-performance **XGBoost** classifier. To achieve interpretability, it integrates **SHAP (SHapley Additive exPlanations)**, applying game theory to quantify the exact impact of every feature on the model's final output.

## Features & Interpretability

* **Imbalanced Data Handling:** Utilizes Synthetic Minority Over-sampling Technique (SMOTE) to prevent the model from biasing toward the majority (non-default) class.
* **Extreme Gradient Boosting:** Hyperparameter-tuned XGBoost classifier optimized for tabular financial data.
* **Global Interpretability:** 
Generates SHAP summary plots highlighting dataset-wide feature importance (e.g., how "Payment Delay" universally impacts risk).
* **Local Interpretability:** 
Generates SHAP force plots explaining specific, individual loan prediction pathways (e.g., why User #402 was flagged for default based on their specific credit limit and age).

## Installation & Usage

# Clone the repository
git clone [https://github.com/yourusername/xai-credit-predictor.git](https://github.com/yourusername/xai-credit-predictor.git)
cd xai-credit-predictor

# Install dependencies
pip install -r requirements.txt

# Run the training and SHAP generation pipeline
python src/train_and_explain.py --data data/UCI_Credit_Card.csv
