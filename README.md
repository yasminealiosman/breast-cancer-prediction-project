

# 🩺 Breast Cancer Prediction Project

## 📌 Project Overview

Breast cancer remains a leading cause of morbidity worldwide. This project applies machine learning techniques to predict breast cancer outcomes using clinical features. Logistic Regression and Gradient Boosting models were trained and evaluated with calibration curves, ROC analysis, and decision curve analysis. Results demonstrate near‑perfect separability between benign and malignant cases, with reliable probability calibration and net clinical benefit. The accompanying Streamlit dashboard provides interactive predictions, dataset upload functionality, and downloadable visualizations, supporting transparency and reproducibility.




## 📂 Dataset Source

This project uses the **Breast Cancer Wisconsin (Diagnostic) dataset**, originally from the  
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic),  
and accessed via **Kaggle**.

- Features: 30 numeric features computed from digitized images of fine needle aspirates (FNAs).  
- Target: Diagnosis (Malignant vs. Benign).  
- Instances: 569 samples.  

The dataset is widely used for benchmarking classification algorithms in healthcare-related machine learning tasks.


## 📂 Repository Structure

```
breast-cancer-prediction/
│
├── dashboard/                # Streamlit dashboard and artifacts
│   ├── artifacts/            # Snapshots of visualizations and tables
│   │   ├── ROC_curve.png
│   │   ├── calibration_curve.png
│   │   └── confusion_matrix.png
│   ├── breast_cancer_dashboard.py
│   ├── breast_cancer_06_dashboard.py
│   ├── X_test.csv
│   └── y_test.csv
│
├── data/                     # Raw and preprocessed datasets
│   ├── raw/
│   │   └── breast_cancer_dataset.csv
│   └── preprocessed/
│       └── breast_cancer_pruned.csv
│
├── models/                   # Serialized models, thresholds, and test sets
│   ├── lr_pipeline.pkl
│   ├── gb_pipeline.pkl
│   ├── threshold_lr.pkl
│   ├── threshold_gb.pkl
│   └── test_set.pkl
│
├── notebooks/                # Jupyter notebooks for workflow stages
│   ├── breast-cancer-01_download-dataset.ipynb
│   ├── breast-cancer-02-exploratory-data-analysis.ipynb
│   ├── breast-cancer-03-preprocessing.ipynb
│   ├── breast-cancer-04-modeling.ipynb
│   ├── breast-cancer-05-reporting.ipynb
│   └── breast-cancer_06_dashboard.ipynb
│
├── .gitignore                # Ignore large files and shortcuts
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies
```


## 🧹 Data Preprocessing

The dataset was prepared with the following steps to ensure reproducibility and interpretability:

- **Data Cleaning**  
  Checked for duplicates and missing values (none present).

- **Feature Scaling**  
  Standardized numeric features for comparability.

- **Feature Engineering**  
  - Ratios (e.g., `perimeter_radius_ratio`) to highlight proportional relationships  
  - Squared terms for non-linear effects  
  - Normalized features to emphasize relative variation  
  - Interaction terms to capture clinically meaningful feature interactions  

- **Pruning**  
  Applied **Variance Inflation Factor (VIF)** analysis to reduce collinearity, retaining engineered features that preserve predictive signal.

- **Train/Test Split**  
  Divided into training and testing sets (e.g., 80/20 split).




## ⚙️ Modeling Approach

To balance interpretability and predictive performance, we applied the following modeling strategies:

- **Algorithms**
  - Logistic Regression: chosen for transparency and clinical interpretability
  - Gradient Boosting: used to capture complex, non-linear relationships

- **Workflow**
  - Models trained on the engineered and pruned feature set
  - Hyperparameter tuning performed with cross-validation
  - Evaluation conducted on a held-out test set

- **Interpretability**
  - Logistic Regression coefficients examined for clinical meaning
  - Gradient Boosting feature importance analyzed to highlight key predictors


## 📈 Evaluation Metrics

Model performance was assessed using multiple metrics to balance accuracy with clinical interpretability:

- **ROC Curves**  
  Compared models on sensitivity vs. specificity trade-offs.

- **AUC (Area Under Curve)**  
  Quantified overall discriminative ability.

- **Calibration Plots**  
  Checked how well predicted probabilities aligned with actual outcomes.

- **Confusion Matrix**  
  Summarized correct vs. incorrect classifications for malignant and benign cases.

- **Decision Curve Analysis**  
  Evaluated clinical usefulness by considering net benefit across threshold probabilities.

---

**Interpretability Focus:**  
- Logistic Regression coefficients were examined for clinical meaning.  
- Gradient Boosting feature importance highlighted key predictors influencing malignancy risk.


## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yasminealiosman/breast-cancer-prediction-project.git
cd breast-cancer-prediction-project
pip install -r requirements.txt
```

---

##  Usage

### 1. Run Notebooks
- `notebooks/breast-cancer-01_download-dataset.ipynb`   
- `notebooks/breast-cancer-02-exploratory-data-analysis.ipynb`  → Exploratory analysis (PCA, separability, class balance checks)   
- `notebooks/breast-cancer-03-preprocessing.ipynb` 
- `notebooks/breast-cancer-04-modeling.ipynb` → Model training, threshold tuning, evaluation
- `notebooks/breast-cancer-05-reporting.ipynb` (evaluation)
- `notebooks/breast-cancer_06_dashboard.ipynb` 


### 2. Launch Dashboard Locally
Run the Streamlit app:
```bash
streamlit run notebooks/breast_cancer_06_dashboard.py
```

The dashboard supports:
- **Batch scoring**: Upload CSVs of patient data (preprocessed format)  
- **Interactive prediction**: Enter single patient features  
- **Artifacts management**: Download ROC curves, confusion matrices, calibration plots, and tuned thresholds  

---

## 🌐 Deployment

This project is deployed on **Streamlit Cloud** for easy access and sharing.  

🔗 **Live Dashboard Preview:** [Breast Cancer Prediction Dashboard](https://breast-cancer-prediction-project-xlaymqx3l7jvnhhhsvjbh8.streamlit.app)

### Steps to Deploy Yourself:
1. Push the repo to GitHub.  
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and connect your GitHub repository.  
3. Select `notebooks/breast_cancer_dashboard.py` as the entry point.  
4. Streamlit Cloud will automatically install dependencies from `requirements.txt` and launch the app.  

---
 
## 📊 Features
- Logistic Regression and Gradient Boosting models  
- Tuned thresholds for optimal F1 and clinical balance  
- Calibration curves for probability reliability  
- Decision curve analysis for net benefit evaluation  
- Feature importance plots for interpretability (LR coefficients as risk factors, GB relative importance)  
- Import new datasets directly into the dashboard for evaluation or prediction  
- Snapshots of visualizations and tables (ROC curves, confusion matrices, calibration plots, DCA results) available for download  



## 🏥 Clinical Trust Notes
- **Accuracy range (99–100%)**: May reflect PCA‑observed separability, stratified train/test splits, and small test size.  
- **ROC curves**: Show near‑perfect separability between benign and malignant cases.  
- **Calibration curves**: Demonstrate probability reliability — predicted risks align with observed outcomes.  
- **Decision curve analysis (DCA)**: Confirms clinical utility by showing net benefit compared to “Treat All” or “Treat None.”  
- **Interpretability**: LR coefficients act as risk factors; GB highlights feature importance consistent with pathology markers.  

---

## 👩🏽‍💻 Author
**Yasmine Ali-Osman**  
- GitHub: [@yasminealiosman](https://github.com/yasminealiosman)  
- LinkedIn: [Yasmine Ali-Osman](https://linkedin.com/in/yasmine-ali-osman-043241206)  

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
```








