"""
Main runner script for complete model evaluation and graph generation
Run this script to generate all results for your paper
"""

import sys
import os
import logging
from pathlib import Path

sys.path.append(os.getcwd())

from backend.core.config import settings
from backend.scripts.comprehensive_model_evaluation import ComprehensiveModelEvaluator
from backend.scripts.generate_paper_reports import ReportGenerator, generate_statistical_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Execute complete evaluation pipeline"""
    
    print("\n" + "=" * 100)
    print(" " * 20 + "INVESTIQ MODEL EVALUATION FOR ACADEMIC PAPER")
    print("=" * 100 + "\n")
    
    print("STEP 1: Running Comprehensive Model Evaluation...")
    print("-" * 100)
    try:
        evaluator = ComprehensiveModelEvaluator()
        evaluator.run_all_evaluations()
        print("✓ Evaluation completed successfully!\n")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        print(f"✗ Evaluation failed: {e}\n")
        return False
    
    print("STEP 2: Generating CSV Reports and LaTeX Tables...")
    print("-" * 100)
    try:
        generator = ReportGenerator()
        generator.generate_csv_reports()
        generator.create_instructions_file()
        print("✓ CSV reports generated successfully!\n")
    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        print(f"✗ Report generation failed: {e}\n")
        return False
    
    print("STEP 3: Creating Statistical Analysis...")
    print("-" * 100)
    try:
        generate_statistical_summary()
        print("✓ Statistical analysis completed!\n")
    except Exception as e:
        logger.error(f"Error generating statistics: {e}")
        print(f"✗ Statistical analysis failed: {e}\n")
        return False
    
    # Display results location
    results_dir = os.path.join(settings.MODEL_DIR, 'evaluation_results')
    
    print("\n" + "=" * 100)
    print("EVALUATION COMPLETE!")
    print("=" * 100)
    print(f"\nResults Location: {results_dir}\n")
    
    print("Generated Files:")
    print("-" * 100)
    
    # List all generated files
    if os.path.exists(results_dir):
        files = sorted(os.listdir(results_dir))
        
        print("\nGraphs (for paper figures):")
        for f in files:
            if f.endswith('.png'):
                print(f"  ✓ {f}")
        
        print("\nReports:")
        for f in files:
            if f.endswith(('.txt', '.csv', '.md', '.tex')):
                print(f"  ✓ {f}")
    
    print("\n" + "=" * 100)
    print("Next Steps for Your Paper:")
    print("=" * 100)
    print("""
1. COPY GRAPHS:
   - Copy PNG files directly into your paper's figures section
   - All graphs are publication-quality (300 DPI)
   - Each graph has a descriptive title and legend

2. USE METRICS:
   - Reference metrics from comprehensive_evaluation_report.txt
   - Copy tables from detailed_comparison.csv for your results section
   - Use statistical_analysis.txt for interpretation guidance

3. ADD TO PAPER:
   - Include LaTeX tables from paper_tables.tex in your manuscript
   - Reference the README.md for detailed descriptions
   - Cite model architectures and metrics used

4. TECHNICAL DETAILS:
   - LSTM: Bidirectional LSTM with Attention (128 hidden, 2 layers)
   - Transformer: Multi-head attention (64 d_model, 4 heads, 2 layers)
   - XGBoost: Gradient boosting classifier (500 estimators)
   - Ensemble: Weighted average (40% LSTM + 60% Transformer)

5. DATA CHARACTERISTICS:
   - Sequence length: 60 trading days
   - Forecast horizon: 7 days
   - Tickers: HDFCBANK, ICICIBANK, INFY, RELIANCE, TCS
   - Time period: 2015-2024 historical data
""")
    print("=" * 100 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
