"""
Churn Prediction (90-Day Window)
Brukd AI-Driven Customer Segmentation & Predictive Engagement

This script builds a predictive model to identify customers at risk 
of churning in the next 90 days for proactive retention campaigns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, precision_recall_curve, f1_score)
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

def load_data():
    """Load segmented data"""
    print("=" * 80)
    print("Loading data for churn prediction...")
    
    df = pd.read_csv('data/data_segmented.csv')
    df_encoded = pd.read_csv('data/data_encoded_segmented.csv')
    
    print(f"✓ Loaded {len(df):,} customers")
    print(f"  Features available: {df_encoded.shape[1]}")
    print(f"\n  Churn Distribution:")
    print(f"    High Risk (Churn=1): {df['Churn_Risk'].sum():,} ({df['Churn_Risk'].mean()*100:.1f}%)")
    print(f"    Low Risk (Churn=0): {(1-df['Churn_Risk']).sum():,} ({(1-df['Churn_Risk'].mean())*100:.1f}%)")
    
    return df, df_encoded

def prepare_features(df_encoded):
    """Prepare features and target for modeling"""
    print("\nPreparing features for churn prediction...")
    
    # Exclude non-predictive columns
    exclude_cols = ['Customer ID', 'Item Purchased', 'CLV_Target', 'Churn_Risk', 
                   'Churn_Risk_Score', 'Value_Tier']
    
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    X = df_encoded[feature_cols]
    y = df_encoded['Churn_Risk']
    
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"  Number of features: {X.shape[1]}")
    print(f"  Target variable: Churn_Risk (0=Low Risk, 1=High Risk)")
    
    return X, y, feature_cols

def balance_dataset(X_train, y_train):
    """Apply SMOTE to balance the dataset"""
    print("\nBalancing dataset with SMOTE...")
    print(f"  Before SMOTE:")
    print(f"    Class 0: {(y_train==0).sum():,}")
    print(f"    Class 1: {(y_train==1).sum():,}")
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"  After SMOTE:")
    print(f"    Class 0: {(y_train_balanced==0).sum():,}")
    print(f"    Class 1: {(y_train_balanced==1).sum():,}")
    
    return X_train_balanced, y_train_balanced

def train_churn_models(X_train, y_train, X_test, y_test):
    """Train and compare churn prediction models"""
    print("\nTraining churn prediction models...")
    
    models = {}
    results = {}
    
    # Random Forest
    print("\n  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'f1': f1_score(y_test, rf_pred),
        'auc': roc_auc_score(y_test, rf_proba)
    }
    print(f"    F1-Score: {results['Random Forest']['f1']:.4f}")
    print(f"    AUC-ROC: {results['Random Forest']['auc']:.4f}")
    
    # XGBoost
    print("\n  Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    
    models['XGBoost'] = xgb
    results['XGBoost'] = {
        'predictions': xgb_pred,
        'probabilities': xgb_proba,
        'f1': f1_score(y_test, xgb_pred),
        'auc': roc_auc_score(y_test, xgb_proba)
    }
    print(f"    F1-Score: {results['XGBoost']['f1']:.4f}")
    print(f"    AUC-ROC: {results['XGBoost']['auc']:.4f}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = models[best_model_name]
    
    print(f"\n✓ Best Model: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")
    
    return best_model, best_model_name, results

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    print(f"\n{'='*80}")
    print(f"CHURN PREDICTION MODEL EVALUATION - {model_name}")
    print(f"{'='*80}")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]:,}")
    print(f"  False Positives: {cm[0,1]:,}")
    print(f"  False Negatives: {cm[1,0]:,}")
    print(f"  True Positives:  {cm[1,1]:,}")
    
    # AUC-ROC
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"\n  AUC-ROC Score: {auc_score:.4f}")
    
    return y_pred, y_proba

def visualize_results(y_test, y_pred, y_proba, model_name):
    """Create comprehensive visualizations"""
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['Low Risk', 'High Risk'])
    axes[0, 0].set_title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xlabel('Predicted')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.4f}')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve', fontweight='bold', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    axes[1, 0].plot(recall, precision, linewidth=2)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Churn Probability Distribution
    axes[1, 1].hist(y_proba[y_test==0], bins=50, alpha=0.5, label='Low Risk (Actual)', color='green')
    axes[1, 1].hist(y_proba[y_test==1], bins=50, alpha=0.5, label='High Risk (Actual)', color='red')
    axes[1, 1].set_xlabel('Churn Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Churn Probability Distribution', fontweight='bold', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    
    plt.tight_layout()
    plt.savefig('visualizations/churn_prediction_evaluation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/churn_prediction_evaluation.png")
    plt.close()

def feature_importance_analysis(model, feature_names):
    """Analyze and visualize feature importance"""
    print("\nAnalyzing feature importance...")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20 features
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Features for Churn Prediction', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig('visualizations/churn_feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: visualizations/churn_feature_importance.png")
        plt.close()
        
        print("\nTop 10 Most Important Features:")
        top_10_indices = indices[-10:][::-1]
        for i, idx in enumerate(top_10_indices, 1):
            print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")

def save_churn_predictions(df, model, df_encoded, feature_cols):
    """Save churn predictions for all customers"""
    print("\nGenerating churn predictions for all customers...")
    
    X = df_encoded[feature_cols]
    churn_proba = model.predict_proba(X)[:, 1]
    churn_pred = model.predict(X)
    
    df['Churn_Probability_90d'] = churn_proba
    df['Churn_Prediction_90d'] = churn_pred
    
    # Risk categories
    df['Churn_Risk_Category'] = pd.cut(churn_proba, 
                                        bins=[0, 0.3, 0.6, 1.0],
                                        labels=['Low', 'Medium', 'High'])
    
    print(f"✓ Churn predictions generated:")
    print(f"  Low Risk: {(df['Churn_Risk_Category']=='Low').sum():,} customers")
    print(f"  Medium Risk: {(df['Churn_Risk_Category']=='Medium').sum():,} customers")
    print(f"  High Risk: {(df['Churn_Risk_Category']=='High').sum():,} customers")
    
    # Save
    df.to_csv('data/data_with_churn_predictions.csv', index=False)
    print("\n✓ Saved: data/data_with_churn_predictions.csv")
    
    return df

def save_model(model, model_name):
    """Save the trained model"""
    joblib.dump(model, 'models/churn_model.pkl')
    print(f"\n✓ Saved model: models/churn_model.pkl ({model_name})")

def main():
    """Main execution function"""
    print("=" * 80)
    print(" " * 20 + "BRUKD AI-DRIVEN CUSTOMER ENGAGEMENT")
    print(" " * 25 + "90-Day Churn Prediction")
    print("=" * 80)
    
    # Load data
    df, df_encoded = load_data()
    
    # Prepare features
    X, y, feature_cols = prepare_features(df_encoded)
    
    # Train-test split
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                         random_state=42, stratify=y)
    print(f"✓ Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    
    # Balance training data
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
    
    # Train models
    best_model, model_name, results = train_churn_models(X_train_balanced, 
                                                          y_train_balanced,
                                                          X_test, y_test)
    
    # Evaluate best model
    y_pred, y_proba = evaluate_model(best_model, X_test, y_test, model_name)
    
    # Visualize results
    visualize_results(y_test, y_pred, y_proba, model_name)
    
    # Feature importance
    feature_importance_analysis(best_model, feature_cols)
    
    # Save predictions for all customers
    df = save_churn_predictions(df, best_model, df_encoded, feature_cols)
    
    # Save model
    save_model(best_model, model_name)
    
    print("\n" + "=" * 80)
    print("✓ Churn prediction modeling complete!")
    print("=" * 80)
    
    return best_model, df

if __name__ == "__main__":
    model, df = main()

