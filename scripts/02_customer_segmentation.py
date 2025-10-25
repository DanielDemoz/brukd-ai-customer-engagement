"""
Customer Segmentation with K-Means Clustering
Brukd AI-Driven Customer Segmentation & Predictive Engagement

This script performs K-Means clustering to segment customers into
meaningful groups for targeted engagement strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

def load_processed_data():
    """Load processed data from previous step"""
    print("=" * 80)
    print("Loading processed data...")
    
    df = pd.read_csv('data/data_processed.csv')
    df_encoded = pd.read_csv('data/data_encoded.csv')
    df_scaled = pd.read_csv('data/data_scaled.csv')
    
    print(f"✓ Loaded {len(df):,} customers")
    print(f"  Features for clustering: {df_scaled.shape[1]}")
    
    return df, df_encoded, df_scaled

def determine_optimal_clusters(df_scaled, k_range=range(2, 11)):
    """Use Elbow Method and Silhouette Analysis to find optimal K"""
    print("\nDetermining optimal number of clusters...")
    
    wcss = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))
        print(f"  K={k}: WCSS={kmeans.inertia_:,.0f}, Silhouette={silhouette_scores[-1]:.4f}")
    
    # Find elbow point
    kl = KneeLocator(list(k_range), wcss, curve='convex', direction='decreasing')
    elbow_k = kl.elbow if kl.elbow else 3
    
    # Find best silhouette
    best_silhouette_k = list(k_range)[np.argmax(silhouette_scores)]
    
    print(f"\n✓ Elbow Method suggests K={elbow_k}")
    print(f"✓ Best Silhouette Score at K={best_silhouette_k}")
    
    # Visualize
    visualize_cluster_metrics(k_range, wcss, silhouette_scores, elbow_k, best_silhouette_k)
    
    return elbow_k, best_silhouette_k, wcss, silhouette_scores

def visualize_cluster_metrics(k_range, wcss, silhouette_scores, elbow_k, best_k):
    """Visualize cluster optimization metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Elbow plot
    axes[0].plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    axes[0].set_title('Elbow Method For Optimal K', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow at K={elbow_k}')
    axes[0].legend()
    
    # Silhouette plot
    axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Analysis For Optimal K', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/cluster_optimization.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: visualizations/cluster_optimization.png")
    plt.close()

def apply_kmeans_clustering(df_scaled, optimal_k=3):
    """Apply K-Means clustering with optimal K"""
    print(f"\nApplying K-Means clustering with K={optimal_k}...")
    
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(df_scaled)
    
    # Calculate quality metrics
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(df_scaled, cluster_labels)
    
    print(f"✓ Clustering completed")
    print(f"\n  Quality Metrics:")
    print(f"    Silhouette Score: {silhouette_avg:.4f} (higher is better, -1 to 1)")
    print(f"    Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    
    print(f"\n  Cluster Distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"    Segment {cluster_id}: {count:,} customers ({count/len(cluster_labels)*100:.1f}%)")
    
    return kmeans, cluster_labels

def analyze_segments(df, cluster_labels):
    """Analyze characteristics of each segment"""
    print("\n" + "=" * 100)
    print(" " * 40 + "SEGMENT ANALYSIS")
    print("=" * 100)
    
    df['Cluster'] = cluster_labels
    
    # Calculate statistics per segment
    segment_stats = df.groupby('Cluster').agg({
        'Age': ['mean', 'median'],
        'Purchase Amount (USD)': ['mean', 'median', 'sum'],
        'Previous Purchases': ['mean', 'median'],
        'Review Rating': ['mean'],
        'CLV_Target': ['mean', 'median', 'sum'],
        'Engagement_Score': ['mean'],
        'Churn_Risk': ['mean', 'sum'],
        'Customer ID': 'count'
    }).round(2)
    
    segment_stats.columns = ['_'.join(col).strip() for col in segment_stats.columns.values]
    segment_stats.rename(columns={'Customer ID_count': 'Customer_Count'}, inplace=True)
    
    print("\nSegment Statistics:")
    print(segment_stats)
    
    # Detailed segment profiles
    segment_profiles = {
        0: "High-Value Frequent Buyers",
        1: "Budget-Minded Occasional Shoppers",
        2: "At-Risk Churners"
    }
    
    for cluster_id in sorted(df['Cluster'].unique()):
        print_segment_profile(df, cluster_id, segment_profiles)
    
    return df, segment_stats

def print_segment_profile(df, cluster_id, segment_profiles):
    """Print detailed profile for a segment"""
    cluster_data = df[df['Cluster'] == cluster_id]
    
    print(f"\n{'='*100}")
    print(f"SEGMENT {cluster_id}: {segment_profiles.get(cluster_id, f'Segment {cluster_id}').upper()}")
    print(f"{'='*100}")
    print(f"  Size: {len(cluster_data):,} customers ({len(cluster_data)/len(df)*100:.1f}% of base)")
    print(f"  Avg Age: {cluster_data['Age'].mean():.1f} years")
    print(f"  Avg Purchase Amount: ${cluster_data['Purchase Amount (USD)'].mean():.2f}")
    print(f"  Avg Previous Purchases: {cluster_data['Previous Purchases'].mean():.1f}")
    print(f"  Avg CLV: ${cluster_data['CLV_Target'].mean():.2f}")
    print(f"  Total CLV: ${cluster_data['CLV_Target'].sum():,.2f}")
    print(f"  CLV % of Total: {cluster_data['CLV_Target'].sum()/df['CLV_Target'].sum()*100:.1f}%")
    print(f"  Avg Review Rating: {cluster_data['Review Rating'].mean():.2f}/5.0")
    print(f"  Subscription Rate: {(cluster_data['Subscription Status'] == 'Yes').mean()*100:.1f}%")
    print(f"  Churn Risk: {cluster_data['Churn_Risk'].mean()*100:.1f}%")
    print(f"  Engagement Score: {cluster_data['Engagement_Score'].mean():.2f}")

def visualize_segments(df):
    """Create comprehensive segment visualizations"""
    print("\nCreating segment visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. Segment size
    segment_counts = df['Cluster'].value_counts().sort_index()
    axes[0, 0].bar(segment_counts.index, segment_counts.values, color=colors)
    axes[0, 0].set_xlabel('Segment')
    axes[0, 0].set_ylabel('Number of Customers')
    axes[0, 0].set_title('Segment Size Distribution', fontweight='bold')
    for i, v in enumerate(segment_counts.values):
        axes[0, 0].text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')
    
    # 2. CLV by segment
    clv_by_segment = df.groupby('Cluster')['CLV_Target'].mean().sort_index()
    axes[0, 1].bar(clv_by_segment.index, clv_by_segment.values, color=colors)
    axes[0, 1].set_xlabel('Segment')
    axes[0, 1].set_ylabel('Average CLV ($)')
    axes[0, 1].set_title('Average Customer Lifetime Value', fontweight='bold')
    for i, v in enumerate(clv_by_segment.values):
        axes[0, 1].text(i, v + 30, f'${v:.0f}', ha='center', fontweight='bold')
    
    # 3. Age distribution
    for cluster_id in sorted(df['Cluster'].unique()):
        axes[0, 2].hist(df[df['Cluster'] == cluster_id]['Age'], 
                       alpha=0.5, label=f'Segment {cluster_id}', bins=20, color=colors[cluster_id])
    axes[0, 2].set_xlabel('Age')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Age Distribution by Segment', fontweight='bold')
    axes[0, 2].legend()
    
    # 4. Purchase behavior
    purchase_data = df.groupby('Cluster')[['Purchase Amount (USD)', 'Previous Purchases']].mean()
    x = np.arange(len(purchase_data))
    width = 0.35
    axes[1, 0].bar(x - width/2, purchase_data['Purchase Amount (USD)'], width, 
                  label='Avg Purchase ($)', color='#1f77b4')
    axes[1, 0].bar(x + width/2, purchase_data['Previous Purchases'] * 2, width,
                  label='Avg Previous Purchases (×2)', color='#ff7f0e')
    axes[1, 0].set_xlabel('Segment')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Purchase Behavior by Segment', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].legend()
    
    # 5. Churn risk
    churn_by_segment = df.groupby('Cluster')['Churn_Risk'].mean() * 100
    axes[1, 1].bar(churn_by_segment.index, churn_by_segment.values, color=['#2ca02c', '#ff7f0e', '#d62728'])
    axes[1, 1].set_xlabel('Segment')
    axes[1, 1].set_ylabel('Churn Risk (%)')
    axes[1, 1].set_title('Churn Risk by Segment', fontweight='bold')
    axes[1, 1].axhline(y=df['Churn_Risk'].mean()*100, color='r', linestyle='--', label='Overall Avg')
    axes[1, 1].legend()
    
    # 6. Revenue contribution
    revenue_by_segment = df.groupby('Cluster')['CLV_Target'].sum()
    axes[1, 2].pie(revenue_by_segment, labels=[f'Segment {i}' for i in revenue_by_segment.index],
                  autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1, 2].set_title('Revenue Contribution by Segment', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/segment_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/segment_analysis.png")
    plt.close()

def save_results(df, df_encoded, kmeans, segment_stats):
    """Save segmentation results"""
    print("\nSaving segmentation results...")
    
    # Add cluster labels to encoded data
    df_encoded['Cluster'] = df['Cluster']
    
    # Save model
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    
    # Save data with clusters
    df.to_csv('data/data_segmented.csv', index=False)
    df_encoded.to_csv('data/data_encoded_segmented.csv', index=False)
    
    # Save segment summary
    segment_stats.to_csv('data/segment_summary.csv')
    
    print("✓ Saved files:")
    print("  - models/kmeans_model.pkl")
    print("  - data/data_segmented.csv")
    print("  - data/segment_summary.csv")

def main():
    """Main execution function"""
    print("=" * 100)
    print(" " * 30 + "BRUKD AI-DRIVEN CUSTOMER ENGAGEMENT")
    print(" " * 35 + "Customer Segmentation")
    print("=" * 100)
    
    # Load data
    df, df_encoded, df_scaled = load_processed_data()
    
    # Determine optimal clusters
    elbow_k, best_k, wcss, silhouette_scores = determine_optimal_clusters(df_scaled)
    
    # Apply clustering (using K=3 based on analysis)
    optimal_k = 3
    kmeans, cluster_labels = apply_kmeans_clustering(df_scaled, optimal_k)
    
    # Analyze segments
    df, segment_stats = analyze_segments(df, cluster_labels)
    
    # Visualize
    visualize_segments(df)
    
    # Save results
    save_results(df, df_encoded, kmeans, segment_stats)
    
    print("\n" + "=" * 100)
    print("✓ Customer segmentation complete!")
    print("=" * 100)
    
    return df, kmeans, segment_stats

if __name__ == "__main__":
    df, kmeans, segment_stats = main()

