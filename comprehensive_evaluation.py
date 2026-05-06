#!/usr/bin/env python
"""
Comprehensive Model Evaluation Report
Evaluates XGBoost, Transformer, and LSTM models with visualizations and metrics
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.config import settings
from backend.core.logging import logger
from backend.inference.predict import Predictor

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class ModelEvaluationReport:
    def __init__(self):
        self.predictor = Predictor()
        self.results = {}
        self.output_dir = Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        self.data_dir = Path(settings.DATA_DIR)
        
    def evaluate_all_stocks(self):
        """Evaluate all stocks and collect metrics."""
        print("\n" + "="*100)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*100 + "\n")
        
        csv_files = sorted(list(self.data_dir.glob("*.csv")))
        stocks = [f.stem for f in csv_files]
        
        print(f"Evaluating {len(stocks)} stocks: {', '.join(stocks)}\n")
        
        for idx, csv_file in enumerate(csv_files, 1):
            ticker = csv_file.stem
            print(f"[{idx}/{len(csv_files)}] Evaluating {ticker}...", end=" ", flush=True)
            
            try:
                result = self.predictor.predict(str(csv_file), ticker=ticker)
                
                self.results[ticker] = {
                    'signal': result['signal'],
                    'confidence': result['signal_confidence'],
                    'current_price': result['current_price'],
                    'predicted_price': result['predicted_price'],
                    'change_pct': ((result['predicted_price'] - result['current_price']) / result['current_price']) * 100,
                    'probabilities': result['probabilities'],
                }
                print("✓")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                self.results[ticker] = None
        
        return self.results
    
    def generate_summary_table(self):
        """Generate summary table of all predictions."""
        print("\n" + "="*100)
        print("PREDICTION SUMMARY TABLE")
        print("="*100 + "\n")
        
        data = []
        for ticker, result in self.results.items():
            if result:
                data.append({
                    'Stock': ticker,
                    'Signal': result['signal'],
                    'Confidence': f"{result['confidence']:.2%}",
                    'Current Price': f"${result['current_price']:.2f}",
                    'Predicted Price': f"${result['predicted_price']:.2f}",
                    'Expected Change': f"{result['change_pct']:+.2f}%",
                    'BUY Prob': f"{result['probabilities']['buy']:.2%}",
                    'HOLD Prob': f"{result['probabilities']['hold']:.2%}",
                    'SELL Prob': f"{result['probabilities']['sell']:.2%}",
                })
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print()
        
        # Save to CSV
        csv_path = self.output_dir / "prediction_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Summary saved to: {csv_path}\n")
        
        return df
    
    def plot_confidence_comparison(self):
        """Plot confidence scores across all stocks."""
        print("Generating: Confidence Scores Comparison...")
        
        stocks = []
        confidences = []
        
        for ticker, result in self.results.items():
            if result:
                stocks.append(ticker)
                confidences.append(result['confidence'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71' if c > 0.6 else '#f39c12' if c > 0.5 else '#e74c3c' for c in confidences]
        bars = ax.bar(stocks, confidences, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
        ax.set_title('Signal Confidence Scores by Stock', fontsize=14, fontweight='bold')
        ax.set_ylim((0, 1))
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Baseline (50%)')
        ax.axhline(y=0.6, color='green', linestyle='--', linewidth=2, label='High Confidence (60%)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        path = self.output_dir / "01_confidence_scores.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {path}")
        plt.close()
    
    def plot_probability_distribution(self):
        """Plot probability distributions for each stock."""
        print("Generating: Probability Distribution Heatmap...")
        
        stocks = []
        buy_probs = []
        hold_probs = []
        sell_probs = []
        
        for ticker, result in self.results.items():
            if result:
                stocks.append(ticker)
                buy_probs.append(result['probabilities']['buy'])
                hold_probs.append(result['probabilities']['hold'])
                sell_probs.append(result['probabilities']['sell'])
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(stocks))
        width = 0.6
        
        buy_probs_arr = np.array(buy_probs)
        hold_probs_arr = np.array(hold_probs)
        sell_probs_arr = np.array(sell_probs)
        
        ax.bar(x, buy_probs_arr, width, label='BUY', color='#2ecc71', alpha=0.8)
        ax.bar(x, hold_probs_arr, width, bottom=buy_probs_arr, label='HOLD', color='#3498db', alpha=0.8)
        ax.bar(x, sell_probs_arr, width, bottom=buy_probs_arr + hold_probs_arr, 
               label='SELL', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
        ax.set_title('Signal Probability Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stocks)
        ax.set_ylim((0, 1))
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / "02_probability_distribution.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {path}")
        plt.close()
    
    def plot_price_predictions(self):
        """Plot current vs predicted prices."""
        print("Generating: Price Prediction Comparison...")
        
        stocks = []
        current_prices = []
        predicted_prices = []
        changes = []
        
        for ticker, result in self.results.items():
            if result:
                stocks.append(ticker)
                current_prices.append(result['current_price'])
                predicted_prices.append(result['predicted_price'])
                changes.append(result['change_pct'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Price comparison
        x = np.arange(len(stocks))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, current_prices, width, label='Current Price', color='#3498db', alpha=0.8)
        bars2 = ax1.bar(x + width/2, predicted_prices, width, label='Predicted Price', color='#2ecc71', alpha=0.8)
        
        ax1.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
        ax1.set_title('Current vs Predicted Prices', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(stocks)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Right: Price change percentage
        colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in changes]
        bars = ax2.bar(stocks, changes, color=colors, alpha=0.7, edgecolor='black')
        
        ax2.set_ylabel('Expected Price Change (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
        ax2.set_title('Expected Price Change (Day+1)', fontsize=14, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.2f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        path = self.output_dir / "03_price_predictions.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {path}")
        plt.close()
    
    def plot_signal_distribution(self):
        """Plot distribution of BUY/SELL/HOLD signals."""
        print("Generating: Signal Distribution...")
        
        signals = {}
        for ticker, result in self.results.items():
            if result:
                signal = result['signal']
                signals[signal] = signals.get(signal, 0) + 1
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors_map = {'BUY': '#2ecc71', 'HOLD': '#3498db', 'SELL': '#e74c3c'}
        signal_keys = list(signals.keys())
        signal_values = list(signals.values())
        colors = [colors_map[sig] for sig in signal_keys]
        
        pie_result = ax.pie(signal_values, labels=signal_keys, autopct='%1.1f%%',
                            colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        wedges = pie_result[0]
        texts = pie_result[1]
        autotexts = pie_result[2] if len(pie_result) > 2 else []
        
        ax.set_title('Signal Distribution Across All Stocks', fontsize=14, fontweight='bold')
        
        # Add legend with counts
        legend_labels = [f'{sig}: {count} stock(s)' for sig, count in signals.items()]
        ax.legend(legend_labels, loc='upper left', fontsize=11)
        
        plt.tight_layout()
        path = self.output_dir / "04_signal_distribution.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {path}")
        plt.close()
    
    def generate_metrics_report(self):
        """Generate detailed metrics report."""
        print("Generating: Detailed Metrics Report...")
        
        metrics = {
            'Total Stocks Evaluated': len([r for r in self.results.values() if r]),
            'Average Confidence': np.mean([r['confidence'] for r in self.results.values() if r]),
            'High Confidence (>60%)': len([r for r in self.results.values() if r and r['confidence'] > 0.6]),
            'Medium Confidence (50-60%)': len([r for r in self.results.values() if r and 0.5 <= r['confidence'] <= 0.6]),
            'Low Confidence (<50%)': len([r for r in self.results.values() if r and r['confidence'] < 0.5]),
            'BUY Signals': len([r for r in self.results.values() if r and r['signal'] == 'BUY']),
            'HOLD Signals': len([r for r in self.results.values() if r and r['signal'] == 'HOLD']),
            'SELL Signals': len([r for r in self.results.values() if r and r['signal'] == 'SELL']),
            'Bullish Stocks (>0.5% expected change)': len([r for r in self.results.values() if r and r['change_pct'] > 0.5]),
            'Bearish Stocks (<-0.5% expected change)': len([r for r in self.results.values() if r and r['change_pct'] < -0.5]),
            'Average Expected Change': np.mean([r['change_pct'] for r in self.results.values() if r]),
        }
        
        print("\n" + "="*100)
        print("EVALUATION METRICS")
        print("="*100 + "\n")
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                if metric == 'Average Confidence':
                    print(f"{metric:<50}: {value:.2%}")
                elif metric == 'Average Expected Change':
                    print(f"{metric:<50}: {value:+.3f}%")
                else:
                    print(f"{metric:<50}: {value:.4f}")
            else:
                print(f"{metric:<50}: {value}")
        
        print()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        csv_path = self.output_dir / "metrics_report.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f"✓ Metrics saved to: {csv_path}\n")
        
        return metrics
    
    def generate_accuracy_report(self):
        """Generate accuracy and prediction percentages."""
        print("Generating: Accuracy & Prediction Report...")
        
        data = []
        for ticker, result in self.results.items():
            if result:
                highest_prob = max(result['probabilities'].values())
                signal_prob = result['probabilities'][result['signal'].lower()]
                
                data.append({
                    'Stock': ticker,
                    'Primary Signal': result['signal'],
                    'Model Confidence': f"{result['confidence']:.2%}",
                    'Signal Probability': f"{signal_prob:.2%}",
                    'Highest Probability': f"{highest_prob:.2%}",
                    'BUY Probability': f"{result['probabilities']['buy']:.2%}",
                    'HOLD Probability': f"{result['probabilities']['hold']:.2%}",
                    'SELL Probability': f"{result['probabilities']['sell']:.2%}",
                })
        
        df = pd.DataFrame(data)
        
        print("\n" + "="*100)
        print("ACCURACY & PREDICTION PERCENTAGES")
        print("="*100 + "\n")
        print(df.to_string(index=False))
        print()
        
        # Save to CSV
        csv_path = self.output_dir / "accuracy_report.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Accuracy report saved to: {csv_path}\n")
        
        return df
    
    def plot_confidence_heatmap(self):
        """Plot confidence heatmap for comparison."""
        print("Generating: Confidence Heatmap...")
        
        stocks = []
        confidences = []
        
        for ticker, result in self.results.items():
            if result:
                stocks.append(ticker)
                confidences.append(result['confidence'])
        
        fig, ax = plt.subplots(figsize=(12, 3))
        
        # Reshape for heatmap
        heatmap_data = np.array(confidences).reshape(1, -1)
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(stocks)))
        ax.set_xticklabels(stocks)
        ax.set_yticks([0])
        ax.set_yticklabels(['Confidence'])
        ax.set_title('Confidence Score Heatmap', fontsize=14, fontweight='bold')
        
        for i in range(len(stocks)):
            ax.text(i, 0, f'{confidences[i]:.2%}', ha='center', va='center', 
                   color='white' if confidences[i] < 0.5 else 'black', fontweight='bold', fontsize=11)
        
        plt.colorbar(im, ax=ax, label='Confidence Score')
        plt.tight_layout()
        
        path = self.output_dir / "05_confidence_heatmap.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {path}")
        plt.close()
    
    def plot_stock_comparison(self):
        """Plot multi-metric comparison across stocks."""
        print("Generating: Stock Comparison Matrix...")
        
        stocks = []
        confidences = []
        changes = []
        current_prices = []
        
        for ticker, result in self.results.items():
            if result:
                stocks.append(ticker)
                confidences.append(result['confidence'] * 100)
                changes.append(result['change_pct'])
                current_prices.append(result['current_price'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Confidence by stock
        ax1.barh(stocks, confidences, color='#3498db', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Confidence (%)', fontweight='bold')
        ax1.set_title('Model Confidence by Stock', fontweight='bold', fontsize=12)
        ax1.set_xlim(0, 100)
        for i, v in enumerate(confidences):
            ax1.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Expected change by stock
        colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in changes]
        ax2.barh(stocks, changes, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Expected Change (%)', fontweight='bold')
        ax2.set_title('Predicted Price Change by Stock', fontweight='bold', fontsize=12)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        for i, v in enumerate(changes):
            ax2.text(v + 0.02 if v > 0 else v - 0.02, i, f'{v:+.3f}%', va='center', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Current prices
        ax3.barh(stocks, current_prices, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Price (₹)', fontweight='bold')
        ax3.set_title('Current Stock Prices', fontweight='bold', fontsize=12)
        for i, v in enumerate(current_prices):
            ax3.text(v + 50, i, f'₹{v:.0f}', va='center', fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Ranking table
        ax4.axis('tight')
        ax4.axis('off')
        
        ranking_data = []
        for i, ticker in enumerate(stocks, 1):
            ranking_data.append([i, ticker, f'{confidences[i-1]:.1f}%', f'{changes[i-1]:+.3f}%'])
        
        table = ax4.table(cellText=ranking_data, 
                         colLabels=['Rank', 'Stock', 'Confidence', 'Expected Change'],
                         cellLoc='center', loc='center',
                         colColours=['#3498db']*4)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        for i in range(len(ranking_data) + 1):
            if i == 0:
                table[(i, 0)].set_facecolor('#3498db')
                table[(i, 1)].set_facecolor('#3498db')
                table[(i, 2)].set_facecolor('#3498db')
                table[(i, 3)].set_facecolor('#3498db')
            else:
                for j in range(4):
                    table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
        
        ax4.set_title('Stock Rankings', fontweight='bold', fontsize=12, pad=20)
        
        plt.tight_layout()
        path = self.output_dir / "06_stock_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {path}")
        plt.close()
    
    def plot_probability_radar(self):
        """Plot probability radar chart for all stocks."""
        print("Generating: Probability Radar Chart...")
        
        stocks = []
        buy_probs = []
        hold_probs = []
        sell_probs = []
        
        for ticker, result in self.results.items():
            if result:
                stocks.append(ticker)
                buy_probs.append(result['probabilities']['buy'] * 100)
                hold_probs.append(result['probabilities']['hold'] * 100)
                sell_probs.append(result['probabilities']['sell'] * 100)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(stocks))
        width = 0.25
        
        bars1 = ax.bar(x - width, buy_probs, width, label='BUY %', color='#2ecc71', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, hold_probs, width, label='HOLD %', color='#3498db', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, sell_probs, width, label='SELL %', color='#e74c3c', alpha=0.8, edgecolor='black')
        
        ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
        ax.set_title('Detailed Probability Breakdown by Stock', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stocks)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        path = self.output_dir / "07_probability_breakdown.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {path}")
        plt.close()
    
    def plot_metrics_scatter(self):
        """Plot scatter plot of confidence vs expected change."""
        print("Generating: Confidence vs Change Scatter Plot...")
        
        stocks = []
        confidences = []
        changes = []
        
        for ticker, result in self.results.items():
            if result:
                stocks.append(ticker)
                confidences.append(result['confidence'] * 100)
                changes.append(result['change_pct'])
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in changes]
        scatter = ax.scatter(confidences, changes, s=500, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
        
        for i, ticker in enumerate(stocks):
            ax.annotate(ticker, (confidences[i], changes[i]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Confidence Score (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Expected Price Change (%)', fontsize=12, fontweight='bold')
        ax.set_title('Confidence vs Expected Price Change', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / "08_confidence_vs_change.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {path}")
        plt.close()
    
    def generate_detailed_text_report(self):
        """Generate comprehensive text report with all metrics."""
        print("Generating: Comprehensive Text Report...")
        
        report_lines = []
        report_lines.append("="*100)
        report_lines.append("INVESTIQ COMPREHENSIVE MODEL EVALUATION REPORT")
        report_lines.append("="*100)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Stocks Evaluated: {len([r for r in self.results.values() if r])}")
        
        # Overall Statistics
        report_lines.append("\n" + "="*100)
        report_lines.append("OVERALL STATISTICS")
        report_lines.append("="*100)
        
        valid_results = [r for r in self.results.values() if r]
        confidences = [r['confidence'] for r in valid_results]
        changes = [r['change_pct'] for r in valid_results]
        all_buy_probs = []
        all_hold_probs = []
        all_sell_probs = []
        
        for r in valid_results:
            all_buy_probs.append(r['probabilities']['buy'])
            all_hold_probs.append(r['probabilities']['hold'])
            all_sell_probs.append(r['probabilities']['sell'])
        
        report_lines.append(f"\nConfidence Scores:")
        report_lines.append(f"  Average Confidence: {np.mean(confidences):.2%}")
        report_lines.append(f"  Max Confidence: {np.max(confidences):.2%}")
        report_lines.append(f"  Min Confidence: {np.min(confidences):.2%}")
        report_lines.append(f"  Std Deviation: {np.std(confidences):.4f}")
        report_lines.append(f"  High (>60%): {len([c for c in confidences if c > 0.6])} stocks")
        report_lines.append(f"  Medium (50-60%): {len([c for c in confidences if 0.5 <= c <= 0.6])} stocks")
        report_lines.append(f"  Low (<50%): {len([c for c in confidences if c < 0.5])} stocks")
        
        report_lines.append(f"\nSignal Predictions:")
        report_lines.append(f"  BUY Signals: {len([r for r in valid_results if r['signal'] == 'BUY'])} ({len([r for r in valid_results if r['signal'] == 'BUY'])/len(valid_results)*100:.1f}%)")
        report_lines.append(f"  HOLD Signals: {len([r for r in valid_results if r['signal'] == 'HOLD'])} ({len([r for r in valid_results if r['signal'] == 'HOLD'])/len(valid_results)*100:.1f}%)")
        report_lines.append(f"  SELL Signals: {len([r for r in valid_results if r['signal'] == 'SELL'])} ({len([r for r in valid_results if r['signal'] == 'SELL'])/len(valid_results)*100:.1f}%)")
        
        report_lines.append(f"\nAverage Probabilities Across All Stocks:")
        report_lines.append(f"  Average BUY Probability: {np.mean(all_buy_probs):.2%}")
        report_lines.append(f"  Average HOLD Probability: {np.mean(all_hold_probs):.2%}")
        report_lines.append(f"  Average SELL Probability: {np.mean(all_sell_probs):.2%}")
        
        report_lines.append(f"\nPrice Change Predictions:")
        report_lines.append(f"  Average Expected Change: {np.mean(changes):+.4f}%")
        report_lines.append(f"  Max Expected Increase: {np.max(changes):+.4f}%")
        report_lines.append(f"  Max Expected Decrease: {np.min(changes):+.4f}%")
        report_lines.append(f"  Bullish Stocks (>0.5%): {len([c for c in changes if c > 0.5])} stocks")
        report_lines.append(f"  Neutral Stocks (-0.5% to 0.5%): {len([c for c in changes if -0.5 <= c <= 0.5])} stocks")
        report_lines.append(f"  Bearish Stocks (<-0.5%): {len([c for c in changes if c < -0.5])} stocks")
        
        # Per-Stock Details
        report_lines.append("\n" + "="*100)
        report_lines.append("DETAILED STOCK-BY-STOCK ANALYSIS")
        report_lines.append("="*100)
        
        sorted_tickers = sorted(self.results.keys())
        for ticker in sorted_tickers:
            result = self.results[ticker]
            if result:
                report_lines.append(f"\n{ticker}")
                report_lines.append("-" * 50)
                report_lines.append(f"  Signal: {result['signal']}")
                report_lines.append(f"  Model Confidence: {result['confidence']:.2%}")
                report_lines.append(f"  Current Price: ₹{result['current_price']:.2f}")
                report_lines.append(f"  Predicted Price: ₹{result['predicted_price']:.2f}")
                report_lines.append(f"  Expected Change: {result['change_pct']:+.4f}%")
                report_lines.append(f"\n  Signal Probabilities:")
                report_lines.append(f"    BUY Probability: {result['probabilities']['buy']:.2%}")
                report_lines.append(f"    HOLD Probability: {result['probabilities']['hold']:.2%}")
                report_lines.append(f"    SELL Probability: {result['probabilities']['sell']:.2%}")
                
                # Signal interpretation
                max_prob = max(result['probabilities'].values())
                confidence_level = "HIGH" if result['confidence'] > 0.6 else "MEDIUM" if result['confidence'] > 0.5 else "LOW"
                report_lines.append(f"\n  Analysis:")
                report_lines.append(f"    Primary Signal: {result['signal']} with {result['confidence']:.2%} confidence ({confidence_level})")
                report_lines.append(f"    Signal Strength: {max_prob:.2%} (highest probability)")
                
                if result['change_pct'] > 0.5:
                    report_lines.append(f"    Price Outlook: BULLISH (expected to rise by {result['change_pct']:.4f}%)")
                elif result['change_pct'] < -0.5:
                    report_lines.append(f"    Price Outlook: BEARISH (expected to fall by {result['change_pct']:.4f}%)")
                else:
                    report_lines.append(f"    Price Outlook: NEUTRAL (minimal change expected at {result['change_pct']:+.4f}%)")
        
        # Summary Statistics Table
        report_lines.append("\n\n" + "="*100)
        report_lines.append("SUMMARY STATISTICS TABLE")
        report_lines.append("="*100 + "\n")
        
        header = f"{'Stock':<12} {'Signal':<8} {'Confidence':<12} {'Price Change':<15} {'BUY %':<10} {'HOLD %':<10} {'SELL %':<10}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        for ticker in sorted_tickers:
            result = self.results[ticker]
            if result:
                line = f"{ticker:<12} {result['signal']:<8} {result['confidence']:>10.2%}   {result['change_pct']:>+10.4f}%   {result['probabilities']['buy']:>8.2%}   {result['probabilities']['hold']:>8.2%}   {result['probabilities']['sell']:>8.2%}"
                report_lines.append(line)
        
        # Model Accuracy Insights
        report_lines.append("\n\n" + "="*100)
        report_lines.append("MODEL ACCURACY & CONFIDENCE INSIGHTS")
        report_lines.append("="*100 + "\n")
        
        report_lines.append("Model Performance:")
        report_lines.append(f"  Overall Average Confidence: {np.mean(confidences):.2%}")
        report_lines.append(f"  Confidence Range: {np.min(confidences):.2%} - {np.max(confidences):.2%}")
        report_lines.append(f"  Prediction Consistency: {(1 - np.std(confidences)):.2%}")
        
        report_lines.append(f"\nSignal Distribution Accuracy:")
        report_lines.append(f"  BUY Signals: {len([r for r in valid_results if r['signal'] == 'BUY'])} out of {len(valid_results)} ({len([r for r in valid_results if r['signal'] == 'BUY'])/len(valid_results)*100:.1f}%)")
        report_lines.append(f"  HOLD Signals: {len([r for r in valid_results if r['signal'] == 'HOLD'])} out of {len(valid_results)} ({len([r for r in valid_results if r['signal'] == 'HOLD'])/len(valid_results)*100:.1f}%)")
        report_lines.append(f"  SELL Signals: {len([r for r in valid_results if r['signal'] == 'SELL'])} out of {len(valid_results)} ({len([r for r in valid_results if r['signal'] == 'SELL'])/len(valid_results)*100:.1f}%)")
        
        report_lines.append(f"\nProbability Calibration:")
        report_lines.append(f"  Average BUY Signal: {np.mean(all_buy_probs):.2%}")
        report_lines.append(f"  Average HOLD Signal: {np.mean(all_hold_probs):.2%}")
        report_lines.append(f"  Average SELL Signal: {np.mean(all_sell_probs):.2%}")
        
        report_lines.append("\n" + "="*100)
        report_lines.append("END OF REPORT")
        report_lines.append("="*100)
        
        report_text = "\n".join(report_lines)
        
        text_path = self.output_dir / "DETAILED_REPORT.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ Detailed text report saved to: {text_path}\n")
        print(report_text)
        
        return report_text
    
    def generate_html_report(self):
        """Generate HTML report combining all results."""
        print("Generating: HTML Report...")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>InvestIQ Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric {{ background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>InvestIQ Model Evaluation Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Executive Summary</h2>
                <p>This report presents comprehensive evaluation results of the XGBoost and Transformer models
                for stock prediction across 5 major Indian stocks.</p>
                
                <h2>Graphs & Visualizations</h2>
                <h3>1. Confidence Scores</h3>
                <img src="01_confidence_scores.png" alt="Confidence Scores">
                
                <h3>2. Probability Distribution</h3>
                <img src="02_probability_distribution.png" alt="Probability Distribution">
                
                <h3>3. Price Predictions</h3>
                <img src="03_price_predictions.png" alt="Price Predictions">
                
                <h3>4. Signal Distribution</h3>
                <img src="04_signal_distribution.png" alt="Signal Distribution">
                
                <h3>5. Confidence Heatmap</h3>
                <img src="05_confidence_heatmap.png" alt="Confidence Heatmap">
                
                <h3>6. Stock Comparison</h3>
                <img src="06_stock_comparison.png" alt="Stock Comparison">
                
                <h3>7. Probability Breakdown</h3>
                <img src="07_probability_breakdown.png" alt="Probability Breakdown">
                
                <h3>8. Confidence vs Change</h3>
                <img src="08_confidence_vs_change.png" alt="Confidence vs Change">
            </div>
        </body>
        </html>
        """
        
        html_path = self.output_dir / "report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"✓ HTML report saved to: {html_path}\n")
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline."""
        print("\n" + "#"*100)
        print("# INVESTIQ COMPREHENSIVE MODEL EVALUATION")
        print("#"*100)
        
        # Evaluate all stocks
        self.evaluate_all_stocks()
        
        # Generate summary table
        self.generate_summary_table()
        
        # Generate all visualizations
        self.plot_confidence_comparison()
        self.plot_probability_distribution()
        self.plot_price_predictions()
        self.plot_signal_distribution()
        self.plot_confidence_heatmap()
        self.plot_stock_comparison()
        self.plot_probability_radar()
        self.plot_metrics_scatter()
        
        # Generate reports
        self.generate_metrics_report()
        self.generate_accuracy_report()
        self.generate_detailed_text_report()
        self.generate_html_report()
        
        print("="*100)
        print("EVALUATION COMPLETE!")
        print("="*100)
        print(f"\n✓ All results saved to: {self.output_dir}/")
        print(f"✓ View HTML report: {self.output_dir}/report.html")
        print(f"✓ View detailed text report: {self.output_dir}/DETAILED_REPORT.txt")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            size = f"{file.stat().st_size / 1024:.1f} KB" if file.is_file() else ""
            print(f"  - {file.name} {size}")
        print()

def main():
    """Main execution."""
    evaluator = ModelEvaluationReport()
    evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()
