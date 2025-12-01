# loan_default_model
Loan Default Prediction – Model Development

Overview

This project continues the previous Milestone 1 work, where a raw loan dataset was cleaned, validated, and prepared for analysis. The objective of this milestone is to develop, evaluate, and then select predictive models that best classify potential loan defaults at LAPO Microfinance Bank.
Using the cleaned dataset, loan_default_cleaned.csv, we will build baseline and advanced models, subsequently evaluate their performance and choose the best model for risk scoring and credit decisions.

Dataset
•	Name: loan_default_cleaned.csv
•	Source: Derived from Kaggle Loan Default Prediction Dataset
•	Rows: ~255,347
•	Columns: 18
•	Description: Includes borrower demographics, financial attributes, loan details, engineered features, and the target variable (Default) indicating loan default (0 = No Default, 1 = Default).
•	File Location (Local):
C:\Users\wuser\OneDrive - Nexford University\New Folder\MILESTONE 1 ASSIGNMENT BAN 6800\loan_default_cleaned.csv

Libraries and Tools
•	Python (pandas, numpy, scikit-learn) – Data processing, modeling
•	Matplotlib / Seaborn – Data visualization
•	GitHub – Version control and hosting
•	Optional: Power BI – Dashboard creation

Steps in Model Development
1. Linking Milestone 1
•	Loaded loan_default_cleaned.csv
•	Separated features (X) and target (y)
•	Train-test split (80%-20%)
•	Scaled numeric features with StandardScaler
2. Modeling Plan
•	Baseline model: Logistic Regression
•	Advanced models: Random Forest, Gradient Boosting (XGBoost/LightGBM)
•	Metrics for evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix
•	Cross-validation and hyperparameter tuning for model selection
•	Handle class imbalance if required
•	Interpret results using feature importance or SHAP values
3. Baseline Model
•	Logistic Regression was trained and evaluated
•	Predictions and probabilities generated
•	Performance assessed using classification metrics and ROC-AUC
4. Advanced Model
•	Random Forest Classifier was trained
•	Predictions, evaluation metrics, and ROC-AUC score computed
•	Feature importance visualized
5. Model Evaluation
•	Compared models based on ROC-AUC, precision, recall, and F1-score
•	Confusion matrices plotted for visual inspection of model performance
•	Feature importance analyzed to identify key predictors of loan default
6. Final Model Selection
•	Selected the model with highest ROC-AUC and balanced precision–recall
•	Preferred model demonstrates:
o	High predictive power
o	Low misclassification rate
o	Robustness for imbalanced datasets
•	Enables risk scoring, borrower segmentation, and informed lending decisions at LAPO Microfinance Bank

Key Outputs
•	Cleaned and preprocessed dataset: loan_default_cleaned.csv
•	Baseline and advanced predictive models trained
•	Model evaluation metrics and feature importance plots generated
•	Recommended model ready for deployment in business analytics

How to Run This Notebook
1.	Clone or download this repository:
2.	git clone https://github.com/YourUsername/loan-default-analysis.git
3.	Ensure Python 3.x is installed with necessary libraries:
4.	pip install pandas numpy scikit-learn matplotlib seaborn
5.	Open the Jupyter Notebook in your preferred environment (Anaconda, VS Code, Jupyter Lab/Notebook).
6.	Update the file path for loan_default_cleaned.csv if needed.
7.	Execute the notebook sequentially to replicate model training and evaluation results.

