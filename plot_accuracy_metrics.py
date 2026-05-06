#!/usr/bin/env python
"""Extract and visualize model accuracy metrics"""

import matplotlib.pyplot as plt
import numpy as np

# Metrics from the training output (retrain_adaptive_final.py)
metrics_data = {
    'HDFCBANK': {'Accuracy': 0.3382, 'Precision': 0.3119, 'Recall': 0.3382, 'F1': 0.2392},
    'ICICIBANK': {'Accuracy': 0.3022, 'Precision': 0.3054, 'Recall': 0.3022, 'F1': 0.1773},
    'INFY': {'Accuracy': 0.3154, 'Precision': 0.3245, 'Recall': 0.3154, 'F1': 0.2156},
    'RELIANCE': {'Accuracy': 0.3289, 'Precision': 0.3312, 'Recall': 0.3289, 'F1': 0.2445},
    'TCS': {'Accuracy': 0.3456, 'Precision': 0.3401, 'Recall': 0.3456, 'F1': 0.2678}
}

stocks = list(metrics_data.keys())
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('XGBoost Model Performance Metrics - All Stocks', fontsize=16, fontweight='bold')

# 1. Accuracy comparison
ax = axes[0, 0]
accuracies = [metrics_data[stock]['Accuracy'] for stock in stocks]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bars1 = ax.bar(stocks, accuracies, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('Accuracy Score', fontweight='bold')
ax.set_title('Model Accuracy by Stock')
ax.set_ylim(0, 0.5)
ax.grid(axis='y', alpha=0.3)
for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
ax.axhline(y=np.mean(accuracies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.3f}')
ax.legend()

# 2. Precision, Recall, F1 comparison
ax = axes[0, 1]
x = np.arange(len(stocks))
width = 0.25
precisions = [metrics_data[stock]['Precision'] for stock in stocks]
recalls = [metrics_data[stock]['Recall'] for stock in stocks]
f1_scores = [metrics_data[stock]['F1'] for stock in stocks]

bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#2ca02c', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, recalls, width, label='Recall', color='#ff7f0e', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='#d62728', alpha=0.8, edgecolor='black')

ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Precision, Recall, and F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(stocks)
ax.legend()
ax.set_ylim(0, 0.4)
ax.grid(axis='y', alpha=0.3)

# 3. All metrics across stocks
ax = axes[1, 0]
for i, metric_name in enumerate(metrics_names):
    values = [metrics_data[stock][metric_name] for stock in stocks]
    ax.plot(stocks, values, marker='o', markersize=8, linewidth=2, label=metric_name)
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('All Metrics Trend Across Stocks')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0.15, 0.4)

# 4. Average metrics by type
ax = axes[1, 1]
avg_metrics = {}
for metric_name in metrics_names:
    avg_metrics[metric_name] = np.mean([metrics_data[stock][metric_name] for stock in stocks])

bars = ax.bar(list(avg_metrics.keys()), list(avg_metrics.values()), 
              color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8, edgecolor='black')
ax.set_ylabel('Average Score', fontweight='bold')
ax.set_title('Average Metrics Across All Stocks')
ax.set_ylim(0, 0.4)
ax.grid(axis='y', alpha=0.3)
for bar, (metric, value) in zip(bars, avg_metrics.items()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('model_accuracy_metrics.png', dpi=300, bbox_inches='tight')
print("📊 Visualization saved: model_accuracy_metrics.png")

# Print summary table
print("\n" + "="*80)
print("MODEL ACCURACY SUMMARY TABLE")
print("="*80)
print(f"{'Stock':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-"*80)
for stock in stocks:
    metrics = metrics_data[stock]
    print(f"{stock:<12} {metrics['Accuracy']:<12.4f} {metrics['Precision']:<12.4f} {metrics['Recall']:<12.4f} {metrics['F1']:<12.4f}")
print("-"*80)
print(f"{'AVERAGE':<12} {np.mean(accuracies):<12.4f} {np.mean(precisions):<12.4f} {np.mean(recalls):<12.4f} {np.mean(f1_scores):<12.4f}")
print("="*80)

# Analysis
print("\n📈 ANALYSIS:")
print(f"  ✓ Average Accuracy:  {np.mean(accuracies):.2%}")
print(f"  ✓ Average Precision: {np.mean(precisions):.2%}")
print(f"  ✓ Average Recall:    {np.mean(recalls):.2%}")
print(f"  ✓ Average F1 Score:  {np.mean(f1_scores):.2%}")
print(f"\n  Best Performer:  {stocks[np.argmax(accuracies)]} ({max(accuracies):.2%})")
print(f"  Needs Improvement: {stocks[np.argmin(accuracies)]} ({min(accuracies):.2%})")
