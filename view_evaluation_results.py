#!/usr/bin/env python3
"""
View Latest Model Evaluation Results
This script displays the latest accuracy metrics and confusion matrix for the custom KNN model.
"""

import os
import glob
from datetime import datetime

def get_latest_evaluation_files():
    """Get the latest evaluation result files."""
    results_folder = 'model_evaluation_results'
    
    if not os.path.exists(results_folder):
        print(f"[ERROR] Results folder '{results_folder}' not found!")
        print("Run the training process first to generate evaluation results.")
        return None, None, None
    
    # Find latest files
    cm_files = glob.glob(os.path.join(results_folder, "*_confusion_matrix_*.png"))
    metrics_files = glob.glob(os.path.join(results_folder, "*_accuracy_metrics_*.png"))
    report_files = glob.glob(os.path.join(results_folder, "*_detailed_report_*.txt"))
    
    if not cm_files or not metrics_files:
        print("[ERROR] No evaluation results found!")
        print("Run the training process first to generate evaluation results.")
        return None, None, None
    
    # Sort by timestamp and get latest
    cm_files.sort(reverse=True)
    metrics_files.sort(reverse=True)
    report_files.sort(reverse=True)
    
    latest_cm = cm_files[0]
    latest_metrics = metrics_files[0]
    latest_report = report_files[0] if report_files else None
    
    return latest_cm, latest_metrics, latest_report

def display_evaluation_summary():
    """Display a summary of the latest evaluation results."""
    latest_cm, latest_metrics, latest_report = get_latest_evaluation_files()
    
    if not latest_cm or not latest_metrics:
        return
    
    print("=" * 60)
    print("           LATEST MODEL EVALUATION RESULTS")
    print("=" * 60)
    print()
    
    # Extract timestamp from filename
    cm_filename = os.path.basename(latest_cm)
    metrics_filename = os.path.basename(latest_metrics)
    
    # Parse timestamp
    try:
        cm_timestamp = cm_filename.split('_')[-1].replace('.png', '')
        metrics_timestamp = metrics_filename.split('_')[-1].replace('.png', '')
        
        # Convert to readable format
        cm_dt = datetime.strptime(cm_timestamp, "%Y%m%d_%H%M%S")
        metrics_dt = datetime.strptime(metrics_timestamp, "%Y%m%d_%H%M%S")
        
        print(f"📊 Evaluation Date: {cm_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
    except:
        print("📊 Evaluation Date: Recent")
        print()
    
    print("📁 Generated Files:")
    print(f"   • Confusion Matrix: {os.path.basename(latest_cm)}")
    print(f"   • Accuracy Metrics: {os.path.basename(latest_metrics)}")
    if latest_report:
        print(f"   • Detailed Report: {os.path.basename(latest_report)}")
    print()
    
    print("📈 What You Can Show to External Teachers:")
    print("   1. Confusion Matrix - Shows prediction accuracy for each class")
    print("   2. Accuracy Metrics - Overall performance visualization")
    print("   3. Detailed Report - Comprehensive numerical metrics")
    print()
    
    print("🔍 File Locations:")
    print(f"   Results Folder: {os.path.abspath('model_evaluation_results')}")
    print(f"   Confusion Matrix: {os.path.abspath(latest_cm)}")
    print(f"   Accuracy Metrics: {os.path.abspath(latest_metrics)}")
    if latest_report:
        print(f"   Detailed Report: {os.path.abspath(latest_report)}")
    print()
    
    print("💡 Tips for Presentation:")
    print("   • Open the PNG files to show visual results")
    print("   • Use the detailed report for numerical analysis")
    print("   • Explain how your custom KNN algorithm works")
    print("   • Highlight the accuracy improvements over time")
    print()
    
    print("=" * 60)

def open_results_folder():
    """Open the results folder in file explorer."""
    import subprocess
    import sys
    
    results_folder = 'model_evaluation_results'
    
    if not os.path.exists(results_folder):
        print(f"[ERROR] Results folder '{results_folder}' not found!")
        return
    
    try:
        if sys.platform == "win32":
            os.startfile(results_folder)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", results_folder])
        else:  # Linux
            subprocess.run(["xdg-open", results_folder])
        
        print(f"[SUCCESS] Opened results folder: {results_folder}")
        
    except Exception as e:
        print(f"[ERROR] Failed to open folder: {e}")
        print(f"Please manually navigate to: {os.path.abspath(results_folder)}")

def main():
    """Main function."""
    print("Face Recognition Model Evaluation Results Viewer")
    print("=" * 50)
    print()
    
    # Display summary
    display_evaluation_summary()
    
    # Ask if user wants to open folder
    try:
        response = input("Would you like to open the results folder? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            open_results_folder()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    
    print("\n" + "=" * 50)
    print("Evaluation results are ready for your presentation!")
    print("Good luck with your final year project! 🎓")

if __name__ == "__main__":
    main()
