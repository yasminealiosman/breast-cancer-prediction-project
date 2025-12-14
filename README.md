# 🩺 Breast Cancer Prediction Project

## 📌 Project Overview
This project applies machine learning techniques to predict breast cancer outcomes using clinical features. It combines **exploratory data analysis (EDA)**, **feature engineering**, **model training**, **threshold tuning**, and an **interactive Streamlit dashboard** for deployment.  

The goal is not only high accuracy but also **clinical interpretability and trust**.
This is achieved through calibration curves, decision curve analysis, and transparent feature importance plots.



## 📂 Repository Structure
```
breast-cancer-prediction/
│
├── notebooks/                # Jupyter notebooks for EDA, modeling, experiments
│   ├── breast_cancer_01_eda.ipynb
│   ├── breast_cancer_02_modeling.ipynb
│   └── breast_cancer_06_dashboard.py
│
├── models/                   # Serialized models and thresholds
│   ├── lr_pipeline.pkl
│   ├── gb_pipeline.pkl
│   ├── threshold_lr.pkl
│   └── threshold_gb.pkl
│
├── data/                     # Raw and processed datasets
│   ├── breast_cancer.csv
│   └── preprocessed.csv
│
├── dashboard/                # Artifacts for Streamlit (CSV, plots)
│   ├── X_test.csv
│   ├── y_test.csv
│   └── ROC_curve.png
│
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── .gitignore                # Ignore large files and shortcuts
```

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yasminealiosman/breast-cancer-prediction.git
cd breast-cancer-prediction
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run Notebooks
- `notebooks/breast_cancer_01_eda.ipynb` → Exploratory analysis (PCA, separability, class balance checks)  
- `notebooks/breast_cancer_02_modeling.ipynb` → Model training, threshold tuning, evaluation  

### 2. Launch Dashboard
Run the Streamlit app locally:
```bash
streamlit run notebooks/breast_cancer_06_dashboard.py
```

The dashboard supports:
- **Batch scoring**: Upload CSVs of patient data (preprocessed format)  
- **Interactive prediction**: Enter single patient features  
- **Artifacts management**: Download ROC curves, confusion matrices, calibration plots, and tuned thresholds  

---

## 📊 Features
- Logistic Regression and Gradient Boosting models  
- Tuned thresholds for optimal F1 and clinical balance  
- Calibration curves for probability reliability  
- Decision curve analysis for net benefit evaluation  
- SHAP explanations and feature importance plots for interpretability  
- Exportable artifacts for reproducibility  

---

## 🏥 Clinical Trust Notes
- **Accuracy range (99–100%)**: May reflect PCA‑observed separability, stratified train/test splits, and small test size.  
- **ROC curves**: Show near‑perfect separability between benign and malignant cases.  
- **Calibration curves**: Demonstrate probability reliability — predicted risks align with observed outcomes.  
- **Decision curve analysis (DCA)**: Confirms clinical utility by showing net benefit compared to “Treat All” or “Treat None.”  
- **Interpretability**: LR coefficients act as risk factors; GB highlights feature importance consistent with pathology markers.  

---

## 📦 Deployment
This project can be deployed on **Streamlit Cloud**:
1. Push repo to GitHub  
2. Connect Streamlit Cloud to the repo  
3. Select `notebooks/breast_cancer_06_dashboard.py` as the entry point  

---

## 🔬 Next Steps
- **External validation**: Test models on independent datasets to confirm generalizability.  
- **Deployment**: Host dashboard online for collaborators and clinicians.  
- **Reporting**: Publish clinical summary and technical appendix for transparency.  
- **Extensions**: Add subgroup analysis, bias testing, and SHAP visualizations.  

---

## 👩🏽‍💻 Author
**Yasmine Ali-Osman**  
- GitHub: [@yasminealiosman](https://github.com/yasminealiosman)  
- LinkedIn: [Yasmine Ali-Osman](https://linkedin.com/in/yasmine-ali-osman-043241206)  







