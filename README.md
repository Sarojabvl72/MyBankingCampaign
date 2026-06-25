# MyBankingCampaign

This repository contains code and analysis for a banking marketing campaign.

The primary work and narrative for this project is in the Jupyter notebook:

- **prompt_III Coded.ipynb** — a Jupyter notebook that contains the coding, data exploration, feature engineering, model training, evaluation, and the prompt-based analysis recorded as "Prompt_III". View it on GitHub: https://github.com/Sarojabvl72/MyBankingCampaign/blob/main/prompt_III%20Coded.ipynb

Other repository items of interest:

- `CRISP-DM-BANK.pdf` — CRISP-DM process document used for planning the analysis
- `data/` — folder containing datasets used in the notebook

Key findings and summary

- Business objective: predict whether a contacted client will subscribe to a term deposit (target `y = yes`) so the bank can prioritize likely subscribers and reduce unnecessary calls.
- Baseline: a majority-class baseline (predicting "no" for every client) achieves roughly 88.7% accuracy due to class imbalance; raw accuracy can therefore be misleading.
- Models compared: Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Support Vector Machine (SVM) were trained on a small feature set (client demographics). With default settings, models are near the baseline; Logistic Regression and SVM are closest to the baseline, while KNN and Decision Tree perform worse on that feature set.
- Model tuning and selection: a grid search over more realistic features (excluding `duration` to avoid leakage) found a tuned Logistic Regression (with class_weight='balanced' and regularization search over C) produced the best balanced accuracy. While its overall accuracy may be lower than the majority baseline, it improves detection of actual subscribers (higher recall for the positive class) and has better ROC AUC.
- Recommended evaluation: use balanced accuracy, precision/recall or an economics-driven scoring function rather than raw accuracy because false negatives and false positives have different business costs.

How to reproduce and get exact metrics

1. Open `prompt_III Coded.ipynb` in Jupyter Notebook or JupyterLab and run all cells. The notebook prints model metrics (accuracy, balanced accuracy, ROC AUC) and shows the best hyperparameters found during grid search.
2. To run the notebook headless and save outputs to a new notebook you can use:

   jupyter nbconvert --to notebook --execute "prompt_III Coded.ipynb" --output "prompt_III Coded.executed.ipynb"

3. For an environment with the required packages, see `requirements.txt` in the repo.

How to run locally

1. Open `prompt_III Coded.ipynb` with Jupyter Notebook or JupyterLab.
2. (Optional) Create a virtual environment and install dependencies:

   pip install -r requirements.txt

3. Run the notebook cells in order. The notebook contains explanations, the prompts used for analysis, and the code required to reproduce the results.

See the notebook for the detailed code, prompts, and cell outputs (including confusion matrix and coefficient inspection for the tuned logistic regression).
