"""
Data Preparation & Feature Engineering
Brukd AI-Driven Customer Segmentation & Predictive Engagement

This script prepares customer data for segmentation, CLV prediction, 
and churn modeling by creating engineered features and transforming raw data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'data/shopping_behavior_updated.csv'
OUTPUT_DIR = 'data/'

def load_data():
    """Load raw customer data"""
    print("=" * 80)
    print("Loading customer data...")
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {len(df):,} customer records with {df.shape[1]} features")
    return df

def create_clv_target(df):
    """Create Customer Lifetime Value target variable"""
    print("\nCreating CLV target variable...")
    df['CLV_Target'] = df['Purchase Amount (USD)'] * df['Previous Purchases']
    print(f"✓ CLV Statistics:")
    print(f"  Average: ${df['CLV_Target'].mean():,.2f}")
    print(f"  Median: ${df['CLV_Target'].median():,.2f}")
    print(f"  Total: ${df['CLV_Target'].sum():,.2f}")
    return df

def create_engagement_features(df):
    """Create engagement and behavioral features"""
    print("\nCreating engagement features...")
    
    # Frequency proxy
    df['Frequency'] = df['Previous Purchases']
    
    # Monetary value
    df['Monetary'] = df['Purchase Amount (USD)']
    
    # Engagement score
    df['Engagement_Score'] = (
        df['Previous Purchases'] * 0.4 + 
        df['Review Rating'] * 10 * 0.3 +
        (df['Subscription Status'] == 'Yes').astype(int) * 50 * 0.3
    )
    
    # Purchase frequency mapping
    purchase_freq_map = {
        'Weekly': 52,
        'Fortnightly': 26,
        'Bi-Weekly': 26,
        'Monthly': 12,
        'Quarterly': 4,
        'Annually': 1,
        'Every 3 Months': 4
    }
    df['Annual_Purchase_Frequency'] = df['Frequency of Purchases'].map(purchase_freq_map)
    
    print("✓ Created engagement features: Frequency, Monetary, Engagement_Score, Annual_Purchase_Frequency")
    return df

def create_churn_risk_features(df):
    """Create churn risk indicators"""
    print("\nCreating churn risk features...")
    
    df['Churn_Risk_Score'] = 0
    
    # Low previous purchases
    df.loc[df['Previous Purchases'] < df['Previous Purchases'].quantile(0.25), 'Churn_Risk_Score'] += 1
    
    # Low review rating
    df.loc[df['Review Rating'] < 3.0, 'Churn_Risk_Score'] += 1
    
    # No subscription
    df.loc[df['Subscription Status'] == 'No', 'Churn_Risk_Score'] += 1
    
    # Low purchase amount
    df.loc[df['Purchase Amount (USD)'] < df['Purchase Amount (USD)'].quantile(0.25), 'Churn_Risk_Score'] += 1
    
    # Binary churn risk label (risk score >= 3 indicates high risk)
    df['Churn_Risk'] = (df['Churn_Risk_Score'] >= 3).astype(int)
    
    print(f"✓ Churn risk distribution:")
    print(f"  High Risk: {df['Churn_Risk'].sum():,} customers ({df['Churn_Risk'].mean()*100:.1f}%)")
    print(f"  Low Risk: {(1-df['Churn_Risk']).sum():,} customers ({(1-df['Churn_Risk'].mean())*100:.1f}%)")
    
    return df

def create_value_tiers(df):
    """Create customer value tier categories"""
    print("\nCreating customer value tiers...")
    df['Value_Tier'] = pd.qcut(df['CLV_Target'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
    print("✓ Value tier distribution:")
    print(df['Value_Tier'].value_counts().sort_index())
    return df

def encode_categorical_features(df):
    """One-hot encode categorical variables"""
    print("\nEncoding categorical variables...")
    
    df_encoded = df.copy()
    
    categorical_cols = ['Gender', 'Category', 'Location', 'Size', 'Color', 'Season',
                       'Subscription Status', 'Shipping Type', 'Discount Applied',
                       'Promo Code Used', 'Payment Method', 'Frequency of Purchases']
    
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    
    print(f"✓ Encoded {len(categorical_cols)} categorical features")
    print(f"  Original shape: {df.shape}")
    print(f"  Encoded shape: {df_encoded.shape}")
    print(f"  New features created: {df_encoded.shape[1] - df.shape[1]}")
    
    return df_encoded

def scale_features(df_encoded):
    """Scale features for clustering"""
    print("\nScaling features for clustering...")
    
    # Exclude IDs and target variables
    exclude_cols = ['Customer ID', 'Item Purchased', 'CLV_Target', 'Churn_Risk', 
                   'Churn_Risk_Score', 'Value_Tier']
    
    clustering_features = [col for col in df_encoded.columns if col not in exclude_cols]
    
    # Scale features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_encoded[clustering_features]),
        columns=clustering_features
    )
    
    print(f"✓ Scaled {len(clustering_features)} features")
    
    return df_scaled, scaler, clustering_features

def save_processed_data(df, df_encoded, df_scaled, scaler):
    """Save all processed datasets"""
    print("\nSaving processed datasets...")
    
    df.to_csv(OUTPUT_DIR + 'data_processed.csv', index=False)
    df_encoded.to_csv(OUTPUT_DIR + 'data_encoded.csv', index=False)
    df_scaled.to_csv(OUTPUT_DIR + 'data_scaled.csv', index=False)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("✓ Saved files:")
    print("  - data/data_processed.csv")
    print("  - data/data_encoded.csv")
    print("  - data/data_scaled.csv")
    print("  - models/scaler.pkl")

def generate_summary(df):
    """Generate data preparation summary report"""
    print("\n" + "=" * 80)
    print(" " * 25 + "DATA PREPARATION SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Total Customers: {len(df):,}")
    print(f"  Total Features: {df.shape[1]}")
    print(f"  Missing Values: {df.isnull().sum().sum()}")
    
    print(f"\n💰 Customer Value Metrics:")
    print(f"  Average CLV: ${df['CLV_Target'].mean():,.2f}")
    print(f"  Median CLV: ${df['CLV_Target'].median():,.2f}")
    print(f"  Total CLV: ${df['CLV_Target'].sum():,.2f}")
    print(f"  CLV Range: ${df['CLV_Target'].min():,.2f} - ${df['CLV_Target'].max():,.2f}")
    
    print(f"\n🎯 Engagement Metrics:")
    print(f"  Avg Previous Purchases: {df['Previous Purchases'].mean():.2f}")
    print(f"  Avg Review Rating: {df['Review Rating'].mean():.2f}/5.0")
    print(f"  Subscription Rate: {(df['Subscription Status'] == 'Yes').mean()*100:.1f}%")
    print(f"  Avg Engagement Score: {df['Engagement_Score'].mean():.2f}")
    
    print(f"\n⚠️ Churn Risk Analysis:")
    print(f"  High-Risk Customers: {df['Churn_Risk'].sum():,} ({df['Churn_Risk'].mean()*100:.1f}%)")
    print(f"  Low-Risk Customers: {(1-df['Churn_Risk']).sum():,} ({(1-df['Churn_Risk'].mean())*100:.1f}%)")
    
    print(f"\n🎨 Customer Distribution:")
    print(f"  Gender Distribution:")
    for gender, count in df['Gender'].value_counts().items():
        print(f"    {gender}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✓ Data preparation complete! Ready for segmentation and modeling.")
    print("=" * 80)

def main():
    """Main execution function"""
    print("=" * 80)
    print(" " * 15 + "BRUKD AI-DRIVEN CUSTOMER ENGAGEMENT")
    print(" " * 20 + "Data Preparation & Feature Engineering")
    print("=" * 80)
    
    # Load data
    df = load_data()
    
    # Create features
    df = create_clv_target(df)
    df = create_engagement_features(df)
    df = create_churn_risk_features(df)
    df = create_value_tiers(df)
    
    # Encode and scale
    df_encoded = encode_categorical_features(df)
    df_scaled, scaler, clustering_features = scale_features(df_encoded)
    
    # Save processed data
    save_processed_data(df, df_encoded, df_scaled, scaler)
    
    # Generate summary
    generate_summary(df)
    
    print("\n✓ Script completed successfully!")
    
    return df, df_encoded, df_scaled, scaler

if __name__ == "__main__":
    df, df_encoded, df_scaled, scaler = main()

