#  Customer Churn Prediction for Telecom Companies

Classify telecom customers as churners or non-churners using usage behavior and demographics.
This end-to-end ML project uses the Telco Customer Churn dataset, applies SMOTE for class imbalance, trains XGBoost and Random Forest models (alongside LR, SVM, LightGBM), and performs **SHAP analysis** for model explainability.
The dashboard outputs actionable customer retention insights backed by model predictions.
# Project Overview

Customer churn is one of the biggest challenges in the telecom industry. 
This project builds a Machine Learning pipeline to:

- **Analyze** customer behavior and identify churn drivers
- **Predict** which customers are likely to churn
- **Visualize** insights through an interactive Streamlit dashboard
- **Recommend** retention strategies based on risk scores


  ### Dataset

- **Source:** [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Features:** 21 columns (demographics, services, account info)
- **Target:** Churn (Yes/No)  ~26.5% churn rate

##  Models Used

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosted decision trees |
| LightGBM | Fast gradient boosting framework |
| SVM | Support vector machine with RBF kernel |

### Handling Class Imbalance
- **SMOTE** (Synthetic Minority Oversampling Technique) is applied to balance the training set

### Model Explainability
- **SHAP** (SHapley Additive exPlanations) analysis using TreeExplainer for XGBoost/Random Forest
- Global feature importance, individual customer explanations, feature dependence plots
  

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix, ROC Curve, Feature Importance

---

##  Streamlit Dashboard

The interactive web application includes **6 pages**:

1. **  Overview**  KPI cards, churn distribution, business insights
2. ** Exploratory Analysis**  Interactive charts, correlation heatmaps
3. ** Predict Churn**  Real-time prediction with risk gauge
4. ** Model Performance**  Metrics comparison, ROC curves, confusion matrices
5. **  SHAP Explainability**  Global/local SHAP analysis, beeswarm plots, feature dependence
6. ** Retention Insights**  SHAP-backed business strategies, executive summary, priority playbook

---

##  Key Findings

- **Month-to-month contracts** have 3x higher churn than yearly contracts
- **Fiber optic customers** churn more than DSL users
- **Electronic check** payment method correlates with higher churn
- **New customers (0-12 months)** are the most vulnerable segment
- Add-on services (security, backup, support) significantly reduce churn

##  Technologies

- **Python 3.8+**
- **Pandas, NumPy** â€” Data manipulation
- **Scikit-learn** â€” Model training & evaluation
- **XGBoost, LightGBM** â€” Advanced ML models
- **SHAP**  Model explainability (Shapley values)
- **Imbalanced-learn**  SMOTE oversampling
- **Plotly** Interactive visualizations
- **Streamlit**  Web application framework
- **Statsmodels**  Statistical analysis


##  License

This project is developed as a capstone project for Predictive Analytics coursework.

---

##  Author

**Capstone Project**  Predictive Analytics, Semester 2
