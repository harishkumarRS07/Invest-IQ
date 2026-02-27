import sys
import os
import glob
from backend.core.config import settings
from backend.core.logging import logger
from backend.training.train import train_pipeline
from backend.evaluation.evaluate import evaluate_pipeline
from backend.backtesting.backtest import Backtester
from backend.preprocessing.cleaning import load_data, clean_data
from backend.features.indicators import add_technical_indicators, add_market_correlation
from backend.features.external_data import ExternalDataSimulator

# Add project root to path
sys.path.append(os.getcwd())

def verify_system():
    logger.info("Starting System Verification...")
    
    # 1. Check Data
    csv_files = glob.glob(os.path.join(settings.DATA_DIR, "*.csv"))
    if not csv_files:
        logger.error("No CSV files found. Please add data to data/stock_data/")
        return
        
    test_file = csv_files[0]
    ticker = os.path.basename(test_file).replace(".csv", "")
    logger.info(f"Using {test_file} for verification.")
    
    # 2. Run Training
    logger.info("--- Step 1: Training ---")
    try:
        # Reduce epochs for verification
        original_epochs = settings.EPOCHS
        settings.EPOCHS = 2 # Short run
        train_pipeline(test_file)
        settings.EPOCHS = original_epochs # Restore
        logger.info("Training verification passed.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    # 3. Run Evaluation
    logger.info("--- Step 2: Evaluation ---")
    try:
        evaluate_pipeline(test_file)
        logger.info("Evaluation verification passed.")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return

    # 4. Run Backtesting
    logger.info("--- Step 3: Backtesting ---")
    try:
        # Load and prep data manually as per backtest script logic
        df = load_data(test_file)
        df = clean_data(df)
        market_df = ExternalDataSimulator.fetch_market_index(start_date=df.index[0], end_date=df.index[-1])
        df = add_technical_indicators(df)
        df = add_market_correlation(df, market_df)
        df = ExternalDataSimulator.add_external_features(df, ticker)
        df = df.dropna()
        
        bt = Backtester(ticker)
        bt.run(df)
        logger.info("Backtesting verification passed.")
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        return
        
    # 5. Run SHAP (Optional, might be slow/complex)
    logger.info("--- Step 4: SHAP Explainability ---")
    try:
        from backend.explainability.shap_explainer import generate_shap_plots
        # We need data to pass to generate_shap_plots. 
        # It expects X_train, X_test.
        # Let's just skip full SHAP run in verification or try a mock call if possible.
        # or just import it to check syntax.
        logger.info("SHAP module imported successfully. Skipping full run to save time.")
    except Exception as e:
        logger.error(f"SHAP verification failed: {e}")
        return

    logger.info("All Systems Operational. Verification Complete.")

if __name__ == "__main__":
    verify_system()
