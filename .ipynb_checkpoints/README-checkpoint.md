\# Breast Cancer Prediction Project



\## 📌 Project Overview

This project applies machine learning techniques to predict breast cancer outcomes using clinical features. It includes exploratory data analysis (EDA), model training, threshold tuning, and an interactive Streamlit dashboard for deployment.



---



\## 📂 Repository Structure

```

breast-cancer-prediction/

│

├── notebooks/                # Jupyter notebooks for EDA, modeling, experiments

│   ├── breast\_cancer\_01\_eda.ipynb

│   ├── breast\_cancer\_02\_modeling.ipynb

│   └── breast\_cancer\_06\_dashboard.py

│

├── models/                   # Serialized models and thresholds

│   ├── model\_lr.pkl

│   ├── model\_gb.pkl

│   ├── threshold\_lr.pkl

│   └── threshold\_gb.pkl

│

├── data/                     # Raw and processed datasets

│   ├── breast\_cancer.csv

│   └── processed\_data.csv

│

├── dashboard/                # Artifacts for Streamlit (CSV, plots)

│   ├── X\_test.csv

│   ├── y\_test.csv

│   └── ROC\_curve.png

│

├── requirements.txt          # Dependencies

├── README.md                 # Project documentation

└── .gitignore                # Ignore large files and shortcuts

```



---



\## ⚙️ Installation

Clone the repository and install dependencies:



```bash

git clone https://github.com/yasminealiosman/breast-cancer-prediction.git

cd breast-cancer-prediction

pip install -r requirements.txt

```



---



\## 🚀 Usage



\### 1. Run Notebooks

Explore data and train models:

\- `notebooks/breast\_cancer\_01\_eda.ipynb` → Exploratory analysis  

\- `notebooks/breast\_cancer\_02\_modeling.ipynb` → Model training and evaluation  



\### 2. Launch Dashboard

Run the Streamlit app locally:

```bash

streamlit run notebooks/breast\_cancer\_06\_dashboard.py

```



The dashboard supports:

\- \*\*Batch scoring\*\*: Upload CSVs of patient data  

\- \*\*Interactive prediction\*\*: Enter single patient features  

\- \*\*Artifacts management\*\*: Download ROC curves, confusion matrices, SHAP plots, and tuned thresholds  



---



\## 📊 Features

\- Logistic Regression and Gradient Boosting models  

\- Tuned thresholds for clinical interpretability  

\- SHAP explanations for feature importance  

\- Exportable artifacts for reproducibility  



---



\## 📦 Deployment

This project can be deployed on \*\*Streamlit Cloud\*\*:

1\. Push repo to GitHub  

2\. Connect Streamlit Cloud to the repo  

3\. Select `notebooks/breast\_cancer\_06\_dashboard.py` as the entry point  



---



\## 👩🏽‍💻 Author

\*\*Yasmine Ali-Osman\*\*  

\- GitHub: \[@yasminealiosman](https://github.com/yasminealiosman)  

\- LinkedIn: \[Yasmine Ali-Osman](https://linkedin.com/in/yasmine-ali-osman-043241206)  







