"""
Utility Functions for Telecom Customer Churn Prediction
========================================================
Helper functions for visualization, metrics, and reporting.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  COLOR PALETTE
# ─────────────────────────────────────────────
COLORS = {
    'primary': '#6C63FF',
    'secondary': '#FF6584',
    'accent': '#00D2FF',
    'success': '#00C9A7',
    'warning': '#FFB800',
    'danger': '#FF4757',
    'dark': '#2D3436',
    'light': '#F8F9FA',
    'gradient': ['#6C63FF', '#FF6584', '#00D2FF', '#00C9A7', '#FFB800'],
    'churn': ['#00C9A7', '#FF4757'],  # No Churn, Churn
}


# ─────────────────────────────────────────────
#  EVALUATION METRICS
# ─────────────────────────────────────────────
def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Calculate and return comprehensive evaluation metrics.
    
    Returns:
        dict with accuracy, precision, recall, f1, roc_auc
    """
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
    }
    
    if y_prob is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def print_classification_report(y_true, y_pred, model_name="Model"):
    """Print formatted classification report."""
    print(f"\n{'='*60}")
    print(f"  📊 Classification Report — {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))


# ─────────────────────────────────────────────
#  VISUALIZATION HELPERS
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name="Model", ax=None):
    """Plot a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='RdPu',
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn'],
        ax=ax, linewidths=0.5, linecolor='white',
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold', pad=15)
    
    return ax


def plot_roc_curves(models_data: dict, ax=None):
    """
    Plot ROC curves for multiple models.
    
    Parameters:
        models_data: dict of {model_name: (y_true, y_prob)}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = COLORS['gradient']
    
    for i, (name, (y_true, y_prob)) in enumerate(models_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5,
                label=f'{name} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_feature_importance(feature_names, importances, model_name="Model", top_n=15, ax=None):
    """Plot top N feature importances."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort and select top N
    indices = np.argsort(importances)[-top_n:]
    
    ax.barh(
        range(len(indices)),
        importances[indices],
        color=COLORS['primary'],
        alpha=0.85,
        edgecolor='white',
        linewidth=0.5
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importances — {model_name}', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, axis='x', alpha=0.3)
    
    return ax


def plot_metrics_comparison(metrics_df: pd.DataFrame, ax=None):
    """Plot a grouped bar chart comparing model metrics."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
    
    x = np.arange(len(metrics_df))
    width = 0.15
    
    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics)/2) * width
        bars = ax.bar(x + offset, metrics_df[metric], width, 
                      label=metric, color=COLORS['gradient'][i % len(COLORS['gradient'])],
                      alpha=0.85, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Model'], fontsize=10)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    
    return ax


# ─────────────────────────────────────────────
#  STYLING
# ─────────────────────────────────────────────
def set_plot_style():
    """Set consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#FAFAFA',
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })


if __name__ == '__main__':
    set_plot_style()
    print("✅ Utils module loaded successfully.")
