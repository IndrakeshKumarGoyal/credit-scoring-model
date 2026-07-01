import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model_pipelines():
    """
    Returns a dictionary of models to train.
    Logistic Regression is wrapped in a pipeline with StandardScaler.
    """
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        ]),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=5, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42, n_estimators=100, max_depth=8, class_weight='balanced', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42, n_estimators=100, learning_rate=0.05, max_depth=4
        )
    }
    return models

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains all models and calculates comprehensive evaluation metrics on the test set.
    """
    models = get_model_pipelines()
    results = {}
    
    for name, model in models.items():
        logging.info(f"Training model: {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability estimates if available (for ROC-AUC and credit scoring)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred.astype(float)
            
        # Calculate standard metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.5
            
        # ROC Curve coordinates
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        # Precision-Recall Curve coordinates
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Extract feature importances if available
        feature_importances = None
        feature_names = X_train.columns.tolist()
        
        if name == 'Logistic Regression':
            # Use coefficients for logistic regression
            importances = np.abs(model.named_steps['model'].coef_[0])
            # Normalize to sum to 1
            if importances.sum() > 0:
                importances = importances / importances.sum()
            feature_importances = dict(zip(feature_names, importances))
        elif hasattr(model, 'feature_importances_'):
            feature_importances = dict(zip(feature_names, model.feature_importances_))
            
        results[name] = {
            'model_object': model,
            'metrics': {
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'ROC-AUC': auc
            },
            'curves': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'precision': precision_vals.tolist(),
                'recall': recall_vals.tolist()
            },
            'confusion_matrix': cm.tolist(),
            'feature_importances': feature_importances
        }
        
        logging.info(f"Model {name} - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        
    return results

def calculate_credit_score(probability):
    """
    Maps a predicted probability of default to a credit score between 300 and 850.
    
    Formula: Score = 300 + 550 * (1 - probability)
    We also return the credit rating band.
    """
    score = int(300 + 550 * (1 - probability))
    
    # Bound score between 300 and 850
    score = max(300, min(850, score))
    
    if score >= 800:
        band = "Exceptional"
        color = "#00c853" # Green
        description = "Excellent credit history. The borrower represents extremely low risk."
    elif score >= 740:
        band = "Very Good"
        color = "#64dd17" # Light Green
        description = "Very clean financial history. Highly likely to be approved."
    elif score >= 670:
        band = "Good"
        color = "#ffd600" # Yellow
        description = "Standard credit risk. Expected approval with standard terms."
    elif score >= 580:
        band = "Fair"
        color = "#ff6d00" # Orange
        description = "Minor credit issues. Approval is possible but may carry higher interest rates."
    else:
        band = "Very Poor"
        color = "#dd2c00" # Red
        description = "Significant credit risk. High likelihood of default. Approve with caution or deny."
        
    return score, band, color, description
