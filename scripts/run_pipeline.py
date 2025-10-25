"""
Complete Pipeline Runner
Brukd AI-Driven Customer Segmentation & Predictive Engagement

This script runs the entire analysis pipeline from start to finish.
"""

import sys
import os
from datetime import datetime
import subprocess

def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 100)
    print(" " * ((100 - len(title)) // 2) + title)
    print("=" * 100 + "\n")

def run_script(script_path, script_name):
    """Run a Python script and handle errors"""
    print_header(f"Running: {script_name}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print(f"✓ {script_name} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"✗ Script not found: {script_path}")
        return False

def main():
    """Main pipeline execution"""
    
    print_header("BRUKD AI-DRIVEN CUSTOMER SEGMENTATION & PREDICTIVE ENGAGEMENT")
    print(" " * 40 + "Complete Analysis Pipeline")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Working directory: {os.getcwd()}\n")
    
    # Define pipeline steps
    pipeline_steps = [
        ("scripts/01_data_preparation.py", "Step 1: Data Preparation & Feature Engineering"),
        ("scripts/02_customer_segmentation.py", "Step 2: Customer Segmentation (K-Means)"),
        ("scripts/03_churn_prediction_90_days.py", "Step 3: 90-Day Churn Prediction"),
        ("scripts/04_engagement_recommendations.py", "Step 4: Engagement Recommendations"),
    ]
    
    # Track results
    results = []
    start_time = datetime.now()
    
    # Run each step
    for script_path, script_name in pipeline_steps:
        step_start = datetime.now()
        success = run_script(script_path, script_name)
        step_duration = (datetime.now() - step_start).total_seconds()
        
        results.append({
            'step': script_name,
            'success': success,
            'duration': step_duration
        })
        
        if not success:
            print_header("PIPELINE FAILED")
            print(f"Failed at: {script_name}")
            print(f"Please check the error messages above and fix any issues.")
            print(f"\nTo resume, run individual scripts starting from the failed step.")
            sys.exit(1)
    
    # Calculate total duration
    total_duration = (datetime.now() - start_time).total_seconds()
    
    # Print summary
    print_header("PIPELINE EXECUTION SUMMARY")
    
    print("All steps completed successfully!\n")
    print("Step-by-step breakdown:\n")
    
    for i, result in enumerate(results, 1):
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{i}. {result['step']}")
        print(f"   Status: {status}")
        print(f"   Duration: {result['duration']:.2f} seconds\n")
    
    print(f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Output files summary
    print("\n" + "-" * 100)
    print("OUTPUT FILES GENERATED:")
    print("-" * 100)
    
    output_files = [
        ("data/data_processed.csv", "Processed customer data with engineered features"),
        ("data/data_encoded.csv", "One-hot encoded features"),
        ("data/data_scaled.csv", "Scaled features for clustering"),
        ("data/data_segmented.csv", "Customer data with segment labels"),
        ("data/data_with_churn_predictions.csv", "Complete dataset with all predictions"),
        ("data/customer_engagement_recommendations.csv", "Personalized recommendations"),
        ("data/high_priority_customers.csv", "High-priority action list"),
        ("data/campaign_templates.csv", "Email/SMS campaign templates"),
        ("data/segment_summary.csv", "Segment statistics summary"),
        ("models/scaler.pkl", "Fitted StandardScaler for production use"),
        ("models/kmeans_model.pkl", "Trained K-Means clustering model"),
        ("models/churn_model.pkl", "Trained churn prediction model"),
        ("visualizations/cluster_optimization.png", "Elbow and silhouette plots"),
        ("visualizations/segment_analysis.png", "Comprehensive segment visualizations"),
        ("visualizations/churn_prediction_evaluation.png", "Churn model performance"),
        ("visualizations/churn_feature_importance.png", "Feature importance for churn"),
    ]
    
    for filepath, description in output_files:
        exists = "✓" if os.path.exists(filepath) else "✗"
        print(f"{exists} {filepath}")
        print(f"  {description}\n")
    
    # Next steps
    print("\n" + "=" * 100)
    print("NEXT STEPS")
    print("=" * 100 + "\n")
    
    print("1. Review the generated visualizations in the 'visualizations/' folder")
    print("2. Examine the segment analysis in 'data/segment_summary.csv'")
    print("3. Check high-priority customers in 'data/high_priority_customers.csv'")
    print("4. Launch the interactive dashboard:")
    print("   streamlit run dashboard_app.py")
    print("5. Read the comprehensive blog post: docs/BLOG_POST.md")
    print("6. Review the main README.md for detailed documentation")
    
    print("\n" + "=" * 100)
    print("✓ PIPELINE COMPLETE! All models trained and ready for deployment.")
    print("=" * 100 + "\n")

if __name__ == "__main__":
    main()

