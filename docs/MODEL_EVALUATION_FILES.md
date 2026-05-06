# 🗂️ Model Evaluation Suite - Navigation Guide

> **Your complete evaluation suite for academic papers is ready!**

---

## 📍 Where to Start

### ⚡ Fastest Way (Recommended)
```
Windows: Double-click → run_model_evaluation.bat
Mac/Linux: python backend/scripts/run_evaluation.py
```
**Time: 5-15 minutes**

---

## 📚 Documentation (Choose Your Depth)

### 1️⃣ Super Quick (2 min read)
📄 **[QUICK_START.md](QUICK_START.md)**
- 3-step quick start
- What you'll get
- Troubleshooting
- **Best for:** Getting started immediately

### 2️⃣ Detailed Guide (15 min read)
📄 **[MODEL_EVALUATION_GUIDE.md](MODEL_EVALUATION_GUIDE.md)**
- Complete technical documentation
- How to use results in your paper
- All metrics explained
- Model architectures
- **Best for:** Understanding everything

### 3️⃣ Setup Summary (10 min read)
📄 **[EVALUATION_SETUP_SUMMARY.md](EVALUATION_SETUP_SUMMARY.md)**
- What was set up
- File manifest
- How to use output
- Checklist for paper
- **Best for:** Knowing what you have

---

## 🎯 Quick Navigation by Task

### I want to...

**Run the evaluation**
→ Execute: `python backend/scripts/run_evaluation.py`
→ Or: Double-click `run_model_evaluation.bat` (Windows)

**Understand the metrics**
→ Read: `statistical_analysis.txt` (in results folder)
→ Or: [MODEL_EVALUATION_GUIDE.md](MODEL_EVALUATION_GUIDE.md) - Metrics section

**Use results in my paper**
→ Copy: PNG files from `evaluation_results/` → Figures
→ Copy: CSV data → Results tables
→ Read: [MODEL_EVALUATION_GUIDE.md](MODEL_EVALUATION_GUIDE.md) - Usage section

**Get detailed prediction plots**
→ Execute: `python backend/scripts/generate_prediction_plots.py`
→ Or: Double-click `run_prediction_plots.bat` (Windows)

**Troubleshoot problems**
→ See: [QUICK_START.md](QUICK_START.md) - Troubleshooting
→ Or: [MODEL_EVALUATION_GUIDE.md](MODEL_EVALUATION_GUIDE.md) - FAQ

**Understand model architectures**
→ See: [MODEL_EVALUATION_GUIDE.md](MODEL_EVALUATION_GUIDE.md) - Technical Details

---

## 📂 File Structure

```
InvestIQ-main/
│
├── 📚 DOCUMENTATION
│   ├── QUICK_START.md                    ← Start here!
│   ├── MODEL_EVALUATION_GUIDE.md         ← Full reference
│   ├── EVALUATION_SETUP_SUMMARY.md       ← What's included
│   └── MODEL_EVALUATION_FILES.md         ← This file
│
├── 🎯 QUICK EXECUTION (Windows)
│   ├── run_model_evaluation.bat          ← Main evaluation
│   └── run_prediction_plots.bat          ← Detailed plots
│
├── 🐍 Python Scripts
│   └── backend/scripts/
│       ├── run_evaluation.py             ← Main orchestrator
│       ├── comprehensive_model_evaluation.py
│       ├── generate_paper_reports.py
│       └── generate_prediction_plots.py
│
└── 📊 Results (generated after running)
    └── backend/models/saved_models/evaluation_results/
        ├── 01_rmse_comparison.png
        ├── 02_r2_comparison.png
        ├── 03_directional_accuracy.png
        ├── 04_xgboost_metrics.png
        ├── 05_performance_heatmap.png
        ├── 06_ticker_performance.png
        ├── 07_ensemble_improvement.png
        ├── 08_box_plots.png
        ├── comprehensive_evaluation_report.txt
        ├── model_summary.csv
        ├── detailed_comparison.csv
        ├── statistical_analysis.txt
        ├── paper_tables.tex
        ├── README.md
        └── prediction_visualizations/ (optional)
```

---

## 🎬 Step-by-Step Workflow

### Phase 1: Setup (1 minute)
- [ ] Ensure Python environment is ready
- [ ] Check that `backend/data/stock_data/` has CSV files
- [ ] Verify models are trained in `backend/models/saved_models/`

### Phase 2: Evaluation (5-15 minutes)
- [ ] Run: `python backend/scripts/run_evaluation.py`
- [ ] Or: Double-click `run_model_evaluation.bat`
- [ ] Wait for "EVALUATION COMPLETE!" message

### Phase 3: Check Results (5 minutes)
- [ ] Open folder: `backend/models/saved_models/evaluation_results/`
- [ ] Review PNG graphs
- [ ] Read `comprehensive_evaluation_report.txt`

### Phase 4: Use in Paper (30+ minutes)
- [ ] Copy PNG files to paper figures folder
- [ ] Copy metrics from CSV to results table
- [ ] Write descriptions and interpretations
- [ ] Add LaTeX tables if needed

### Phase 5: Optional - Details (5-10 minutes)
- [ ] Run: `python backend/scripts/generate_prediction_plots.py`
- [ ] Add prediction visualizations to appendix
- [ ] Include residual analysis

---

## 📊 What Each File Does

### 🎯 Execution Files

**`run_model_evaluation.bat`** (Windows only)
- One-click evaluation runner
- Activates environment
- Runs comprehensive evaluation
- Shows results location
- **Use when:** You want simplest execution

**`run_prediction_plots.bat`** (Windows only)
- One-click prediction plot generator
- Creates detailed visualizations
- **Use when:** You want optional detailed plots

### 🐍 Python Scripts

**`run_evaluation.py`**
- Main orchestrator script
- Runs evaluation, reports, statistics
- Recommended for all systems
- **Use when:** Cross-platform or CLI preferred

**`comprehensive_model_evaluation.py`**
- Core evaluation engine
- Evaluates all 4 models
- Generates 8 graphs
- **Use when:** You want just graphs & metrics

**`generate_paper_reports.py`**
- Generates CSV/LaTeX exports
- Creates statistical analysis
- Generates documentation
- **Use when:** You need formatted reports

**`generate_prediction_plots.py`**
- Creates detailed prediction visualizations
- Generates residual analysis
- **Use when:** You need appendix-ready plots

### 📚 Documentation Files

**`QUICK_START.md`**
- Fastest reference guide
- Key information highlighted
- Common tasks and solutions
- **Read when:** You need quick answers

**`MODEL_EVALUATION_GUIDE.md`**
- Complete technical documentation
- Detailed metric explanations
- How to use in paper
- Architecture details
- **Read when:** You want full understanding

**`EVALUATION_SETUP_SUMMARY.md`**
- What was created and why
- File manifest
- Checklist for paper
- **Read when:** You want setup overview

---

## ⏱️ Time Guide

| Task | Time |
|------|------|
| Read QUICK_START.md | 2 min |
| Read EVALUATION_SETUP_SUMMARY.md | 10 min |
| Run evaluation | 5-15 min |
| Copy results to paper | 15-30 min |
| Read MODEL_EVALUATION_GUIDE.md (full) | 30 min |
| Generate prediction plots (optional) | 5-10 min |
| **Total for paper-ready results** | **30-60 min** |

---

## ✅ Success Checklist

After everything runs, you should have:

- [ ] 8 PNG graph files (300 DPI, publication quality)
- [ ] `comprehensive_evaluation_report.txt` with all metrics
- [ ] `detailed_comparison.csv` with model performance data
- [ ] `statistical_analysis.txt` explaining what metrics mean
- [ ] `paper_tables.tex` ready to include in LaTeX
- [ ] All results in: `backend/models/saved_models/evaluation_results/`

---

## 🔗 Cross-References

### From QUICK_START.md
- Need full details? → See MODEL_EVALUATION_GUIDE.md
- Want to know what was set up? → See EVALUATION_SETUP_SUMMARY.md

### From MODEL_EVALUATION_GUIDE.md
- Want quick answers? → See QUICK_START.md
- Need setup info? → See EVALUATION_SETUP_SUMMARY.md

### From EVALUATION_SETUP_SUMMARY.md
- Want quick start? → See QUICK_START.md
- Want detailed guide? → See MODEL_EVALUATION_GUIDE.md

---

## 🎓 For Your Academic Paper

### What to Cite
"Evaluation was performed using a comprehensive suite comprising:
- 4 machine learning models (LSTM Attention, Transformer, XGBoost, Ensemble)
- 5 Indian stock tickers (HDFCBANK, ICICIBANK, INFY, RELIANCE, TCS)
- 8 performance metrics across all models"

### What to Include
- ✅ 2-3 graphs from `evaluation_results/`
- ✅ Results table with metrics from CSV
- ✅ Model architecture descriptions
- ✅ Metric interpretations from `statistical_analysis.txt`

### What to Append (Optional)
- 📊 Detailed prediction plots
- 📈 Residual analysis
- 📌 Per-ticker breakdowns

---

## 🚀 Get Started Now

### Absolute Quickest
```
Windows: You're reading this → Double-click run_model_evaluation.bat
```

### Two Commands
```bash
python backend/scripts/run_evaluation.py
# Then look in: backend/models/saved_models/evaluation_results/
```

### Or Follow These Docs
1. Read [QUICK_START.md](QUICK_START.md) (2 min)
2. Run evaluation (10 min)
3. Check results (5 min)
4. Use in paper (30 min)

---

## 🆘 Need Help?

**Quick answer:** → [QUICK_START.md](QUICK_START.md) Troubleshooting  
**Detailed help:** → [MODEL_EVALUATION_GUIDE.md](MODEL_EVALUATION_GUIDE.md) FAQ  
**Setup issues:** → [EVALUATION_SETUP_SUMMARY.md](EVALUATION_SETUP_SUMMARY.md) Checklist  

---

## 📝 Summary

You now have:
✅ 4 complete Python scripts for evaluation  
✅ 2 Windows batch files for easy execution  
✅ 3 comprehensive documentation files  
✅ Setup for automatic graph generation  
✅ CSV/LaTeX export capabilities  
✅ Optional detailed visualization scripts  

**Everything you need to complete your paper's evaluation section!**

---

**Next Step:** Read [QUICK_START.md](QUICK_START.md) or run `run_model_evaluation.bat` →

*InvestIQ Model Evaluation Suite - Complete & Ready*
