

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="Breast Cancer Prediction Dashboard", layout="wide")
st.title("Breast Cancer Prediction Dashboard")
st.caption("This dashboard is for research/decision support, not a substitute for medical diagnosis.")

# -------------------------
# Paths (raw strings only)
# -------------------------
MODELS_DIR = r"C:/Users/yasmine/Documents/Portfolio/DataSciencePortfolio/Projects/Breast-Cancer/models"
DASHBOARD_DIR = r"C:/Users/yasmine/Documents/Portfolio/DataSciencePortfolio/Projects/Breast-Cancer/dashboard"



# -------------------------
# Utilities
# -------------------------
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0,0], cm[0,1]
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan

def compute_ppv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[1,1], cm[0,1]
    return tp / (tp + fp) if (tp + fp) > 0 else np.nan

def compute_npv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fn = cm[0,0], cm[1,0]
    return tn / (tn + fn) if (tn + fn) > 0 else np.nan

def compute_brier(y_true, y_prob):
    return brier_score_loss(y_true, y_prob)

def plot_confusion_matrix(y_true, y_pred, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_yticklabels(["Benign", "Malignant"], rotation=0)
    return fig

def plot_roc_curves(curves, title="ROC curves"):
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1], "k--", label="Chance")
    for name, (fpr, tpr, auc, color) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return fig

def decision_curve(y_true, y_prob, thresholds=np.linspace(0.01, 0.99, 50)):
    """
    Net benefit = (TP/n) - (FP/n) * (threshold / (1 - threshold))
    """
    n = len(y_true)
    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        net_benefit = (tp/n) - (fp/n) * (thr / (1 - thr))
        rows.append({"threshold": thr, "net_benefit": net_benefit})
    return pd.DataFrame(rows)


# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Configuration")

# Keep only test data paths
x_test_path = st.sidebar.text_input(
    "X_test CSV (features)",
    r"C:/Users/yasmine/Documents/Portfolio/DataSciencePortfolio/Projects/Breast-Cancer/dashboard/X_test.csv"
)
y_test_path = st.sidebar.text_input(
    "y_test CSV (labels)",
    r"C:/Users/yasmine/Documents/Portfolio/DataSciencePortfolio/Projects/Breast-Cancer/dashboard/y_test.csv"
)





# Model selection
selected_models = st.sidebar.multiselect(
    "Models to include",
    options=["LR", "GB"],
    default=["LR", "GB"]
)

# Threshold tuning sliders
st.sidebar.subheader("Threshold tuning")
threshold_controls = {}
for m in selected_models:
    # Set default slider value per model
    default_val = 0.5
    if m == "LR":
        default_val = 0.45
    elif m == "GB":
        default_val = 0.39

    threshold_controls[m] = st.sidebar.slider(
        f"{m} threshold",
        0.0, 1.0, default_val, 0.01
    )

# Only keep calibration checkbox
show_calibration = st.sidebar.checkbox("Show calibration curves", value=True)


# -------------------------
# Load test data
# -------------------------
try:
    X_test = pd.read_csv(x_test_path)
    y_test_df = pd.read_csv(y_test_path)
    y_test = y_test_df.iloc[:, 0]
    if X_test.empty or y_test.empty:
        st.error("❌ One or both test files are empty. Please check X_test.csv and y_test.csv.")
        st.stop()
    st.success(f"✅ Test data loaded (X_test shape: {X_test.shape}, y_test length: {len(y_test)})")
except Exception as e:
    st.error(f"❌ Failed to load test data: {e}")
    st.stop()



# -------------------------
# Load models + thresholds
# -------------------------
models = {}
thresholds = {}
load_msgs = []

for key in selected_models:
    # Model
    try:
        model_path = rf"{MODELS_DIR}/model_{key.lower()}.pkl"
        model = joblib.load(model_path)
        models[key] = model
        # Threshold
        try:
            thr_path = rf"{MODELS_DIR}/threshold_{key.lower()}.pkl"
            thr = joblib.load(thr_path)
        except Exception:
            thr = threshold_controls.get(key, 0.5)
            st.sidebar.warning(f"⚠️ Threshold file missing for {key}. Using slider/default = {thr:.2f}")
        thresholds[key] = float(thr)
        load_msgs.append(f"✅ {key} loaded (threshold={thresholds[key]:.3f})")
    except Exception as e:
        load_msgs.append(f"⚠️ {key} model file missing or unreadable: {e}")

st.subheader("Model load status")
for msg in load_msgs:
    st.write(msg)
if not models:
    st.error("❌ No models loaded. Please ensure .pkl files exist in models/.")
    st.stop()


# -------------------------
# Metric helper functions
# -------------------------
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, brier_score_loss

def _specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0,0], cm[0,1]
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan

def _ppv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[1,1], cm[0,1]
    return tp / (tp + fp) if (tp + fp) > 0 else np.nan

def _npv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fn = cm[0,0], cm[1,0]
    return tn / (tn + fn) if (tn + fn) > 0 else np.nan

def _compute_threshold_metrics(name, model, X, y, thr):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= thr).astype(int)
    return {
        "Model": name,
        "Threshold": thr,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "Specificity": _specificity(y, y_pred),
        "Accuracy": accuracy_score(y, y_pred),
        "ROC_AUC": roc_auc_score(y, y_prob),
        "PPV": _ppv(y, y_pred),
        "NPV": _npv(y, y_pred),
        "Brier": brier_score_loss(y, y_prob)
    }





# -------------------------
# Overview
# -------------------------
from sklearn.metrics import accuracy_score

st.subheader("Overview")


# Test set size
n_test = len(X_test)

# Models compared
models_list = ", ".join(models.keys()) if models else "—"

# Compute accuracy range directly from models
acc_values = []
for name, model in models.items():
    try:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_values.append(acc * 100)
    except Exception as e:
        st.warning(f"Could not compute accuracy for {name}: {e}")

acc_range_txt = "—"
if acc_values:
    acc_min = min(acc_values)
    acc_max = max(acc_values)
    acc_range_txt = f"{acc_min:.1f}%" if acc_min == acc_max else f"{acc_min:.1f}%–{acc_max:.1f}%"

# Display metrics
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(label="Test set size", value=n_test)
with c2:
    st.metric(label="Models compared", value=models_list)
with c3:
    st.metric(label="Accuracy range", value=acc_range_txt)

# Interpretability note
st.markdown("""
**Interpretation Notes:**
- The very high accuracy range (99–100%) may reflect:
  - **Feature separability:** During exploratory analysis (EDA), PCA showed clear clustering between benign and malignant cases, suggesting the dataset is inherently well-separated.
  - **Class balance handling:** We used stratified train/test splits to preserve class ratios, reducing bias from imbalance.
  - **Test size:** The hold-out test set is relatively small (114 cases), which can inflate performance metrics.
- **ROC curves** confirm separability: both models achieve near-perfect AUC, meaning they can distinguish malignant from benign cases almost flawlessly on this dataset.
- **Calibration curves** demonstrate probability reliability: predicted risks align with observed outcomes, which is essential for clinical trust.
- **Decision curve analysis (DCA)** shows positive net benefit compared to treating all or none, providing evidence of clinical utility.
""")

# -------------------------
# Model comparison table + bar plot (live metrics)
# -------------------------
st.subheader("Model comparison")

metrics_rows = []
for name, model in models.items():
    try:
        if hasattr(model, "predict_proba"):
            thr = thresholds.get(name, 0.5)
            metrics_rows.append(_compute_threshold_metrics(name, model, X_test, y_test, thr))
    except Exception as e:
        st.warning(f"Skipping metrics for {name}: {e}")

if metrics_rows:
    eval_df = pd.DataFrame(metrics_rows, columns=[
        "Model", "Threshold", "Precision", "Recall", "F1", "Specificity",
        "Accuracy", "ROC_AUC", "PPV", "NPV", "Brier"
    ])
    st.dataframe(eval_df[["Model","Precision","Recall","F1","Specificity","ROC_AUC","Accuracy"]],
                 use_container_width=True)

    show_cols = ["Recall","Specificity","F1","ROC_AUC"]
    fig_bar, ax = plt.subplots()
    eval_df.set_index("Model")[show_cols].plot(
        kind="bar", ax=ax,
        color=["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    )
    ax.set_title("Balanced metrics comparison (Recall, Specificity, F1, ROC_AUC)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig_bar)
else:
    st.info("No live metrics available. Ensure models are loaded and support predict_proba.")

# -------------------------
# ROC curves overlay (live)
# -------------------------
st.subheader("ROC curves")
roc_curves = {}
color_map = {"LR": "#1f77b4", "GB": "#ff7f0e"}

for name, model in models.items():
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_curves[name] = (fpr, tpr, auc, color_map.get(name, None))
    except Exception as e:
        st.warning(f"Skipping ROC for {name}: {e}")

if roc_curves:
    fig_roc, ax = plt.subplots()
    ax.plot([0,1],[0,1],"k--",label="Chance")
    for name,(fpr,tpr,auc,color) in roc_curves.items():
        ax.plot(fpr,tpr,label=f"{name} (AUC={auc:.3f})",color=color)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves by model (AUC in legend)")
    ax.legend(loc="lower right")
    st.pyplot(fig_roc)
else:
    st.info("No ROC curves available. Ensure models are loaded and support predict_proba.")

# -------------------------
# Calibration curves (live)
# -------------------------
if show_calibration and models:
    st.subheader("Calibration curves")
    cal_cols = st.columns(min(len(models), 3))
    for idx,(name,model) in enumerate(models.items()):
        try:
            if hasattr(model,"predict_proba"):
                y_prob = model.predict_proba(X_test)[:,1]
                prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

                fig_cal, ax = plt.subplots()
                ax.plot(prob_pred, prob_true, marker='o', label=name, color=color_map.get(name,None))
                ax.plot([0,1],[0,1],"k--",label="Perfectly calibrated")
                ax.set_xlabel("Predicted probability")
                ax.set_ylabel("True probability")
                ax.set_title(f"Calibration Curve - {name}")
                ax.legend()
                with cal_cols[idx % len(cal_cols)]:
                    st.pyplot(fig_cal)
        except Exception as e:
            st.warning(f"Skipping calibration for {name}: {e}")

# -------------------------
# Confusion matrices (live)
# -------------------------
st.subheader("Confusion matrices")
if models:
    cm_cols = st.columns(min(len(models), 4))
    for idx,(name,model) in enumerate(models.items()):
        thr = thresholds.get(name,0.5)
        try:
            if hasattr(model,"predict_proba"):
                proba = model.predict_proba(X_test)[:,1]
                y_pred = (proba >= thr).astype(int)
                fig_cm = plot_confusion_matrix(y_test, y_pred, title=f"{name} (thr={thr:.3f})")
                with cm_cols[idx % len(cm_cols)]:
                    st.pyplot(fig_cm)
        except Exception as e:
            st.warning(f"Skipping CM for {name}: {e}")
else:
    st.info("No models available to generate confusion matrices.")

# -------------------------
# Interpretability (live generation)
# -------------------------
st.header("Interpretability")

def _specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0,0], cm[0,1]
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan

def _ppv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[1,1], cm[0,1]
    return tp / (tp + fp) if (tp + fp) > 0 else np.nan

def _npv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fn = cm[0,0], cm[1,0]
    return tn / (tn + fn) if (tn + fn) > 0 else np.nan

def _compute_threshold_metrics(name, model, X, y, thr):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= thr).astype(int)
    return {
        "Model": name,
        "Threshold": thr,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "Specificity": _specificity(y, y_pred),
        "Accuracy": accuracy_score(y, y_pred),
        "ROC_AUC": roc_auc_score(y, y_prob),
        "PPV": _ppv(y, y_pred),
        "NPV": _npv(y, y_pred),
        "Brier": brier_score_loss(y, y_prob)
    }

st.subheader("Threshold tuning metrics")
metrics_rows = []
for name, model in models.items():
    try:
        if hasattr(model, "predict_proba"):
            thr = thresholds.get(name, 0.5)
            metrics_rows.append(_compute_threshold_metrics(name, model, X_test, y_test, thr))
        else:
            st.warning(f"{name} does not support predict_proba; skipping metrics.")
    except Exception as e:
        st.warning(f"Could not compute metrics for {name}: {e}")

if metrics_rows:
    thr_metrics_df = pd.DataFrame(metrics_rows, columns=[
        "Model", "Threshold", "Precision", "Recall", "F1", "Specificity",
        "Accuracy", "ROC_AUC", "PPV", "NPV", "Brier"
    ])
    st.dataframe(thr_metrics_df, use_container_width=True)
    csv_data = thr_metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download threshold tuning metrics (CSV)",
        data=csv_data,
        file_name="threshold_tuning_metrics.csv",
        mime="text/csv"
    )
else:
    st.info("No threshold metrics available. Ensure models are loaded and support predict_proba.")



st.subheader("Feature importance and risk factors")

for name, model in models.items():
    try:
        # 🔑 Unwrap pipeline if needed
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            base_model = model.named_steps["clf"]
        else:
            base_model = model

        # Logistic Regression coefficients
        if name == "LR" and hasattr(base_model, "coef_"):
            coefs = base_model.coef_[0]
            feat_imp_lr = pd.DataFrame({
                "Feature": X_test.columns,
                "Coefficient": coefs,
                "Impact": ["↑ risk" if c > 0 else "↓ risk" for c in coefs]
            }).sort_values("Coefficient", ascending=False)

            fig_lr, ax = plt.subplots(figsize=(8, max(4, len(feat_imp_lr) * 0.25)))
            sns.barplot(
                x="Coefficient", y="Feature", hue="Impact", data=feat_imp_lr,
                palette={"↑ risk": "darkred", "↓ risk": "steelblue"}, dodge=False, ax=ax
            )
            ax.set_title("LR feature impacts (Positive = higher malignancy risk)")
            ax.legend(title="Impact", loc="best")
            st.pyplot(fig_lr)
            st.dataframe(feat_imp_lr)

        # Gradient Boosting feature importances
        elif name == "GB" and hasattr(base_model, "feature_importances_"):
            importances = base_model.feature_importances_
            feat_imp_gb = pd.DataFrame({
                "Feature": X_test.columns,
                "Importance": importances
            }).sort_values("Importance", ascending=False).head(15)

            fig_gb, ax = plt.subplots(figsize=(8, max(4, len(feat_imp_gb) * 0.25)))
            sns.barplot(x="Importance", y="Feature", data=feat_imp_gb,
                        palette="viridis", dodge=False, ax=ax)
            ax.set_title("GB feature importance (top 15)")
            st.pyplot(fig_gb)
            st.dataframe(feat_imp_gb)

        else:
            st.warning(f"{name} does not expose coefficients or feature importances.")

    except Exception as e:
        st.warning(f"Skipping interpretability for {name}: {e}")


st.markdown("""
**Interpretability notes:**
- Logistic Regression (LR) coefficients act as risk factors: positive values increase malignancy risk, negative values decrease it.
- Gradient Boosting (GB) provides relative feature importance scores; calibration curves assess probability reliability.
- Threshold tuning metrics (Precision, Recall, Specificity, PPV, NPV, Brier) are generated live from the test set at your selected thresholds.
""")




# -------------------------
# Interactive prediction
# -------------------------
st.subheader("Interactive prediction")

if X_test is not None and not X_test.empty and models:
    feature_cols = list(X_test.columns)
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_test[c])]

    # Option to use all or top 15 features
    mode_choice = st.radio("Select input mode", ["All features", "Top 15 features"], index=0)

    if mode_choice == "Top 15 features":
        # For simplicity, use variance or importance ranking if available
        # Here we just take first 15 numeric columns
        use_cols = numeric_cols[:15] if len(numeric_cols) >= 15 else numeric_cols
    else:
        use_cols = numeric_cols

    with st.form("single_prediction"):
        inputs = {}
        for c in use_cols:
            try:
                default_val = float(np.nanmean(X_test[c])) if c in X_test.columns else 0.0
            except Exception:
                default_val = 0.0
            inputs[c] = st.number_input(c, value=default_val)
        model_choice = st.selectbox("Model", options=list(models.keys()), key="single_model")
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            # Build full input row with all features
            df_in = pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols)
            for feat, val in inputs.items():
                if feat in df_in.columns:
                    df_in.at[0, feat] = val

            model = models[model_choice]
            thr = thresholds.get(model_choice, 0.5)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df_in)[:, 1][0]
                pred = int(proba >= thr)

                y_prob_test = model.predict_proba(X_test)[:, 1]
                y_pred_test = (y_prob_test >= thr).astype(int)
                ppv = _ppv(y_test, y_pred_test)
                npv = _npv(y_test, y_pred_test)
                brier = brier_score_loss(y_test, y_prob_test)

                st.success(f"Probability: {proba:.3f} | Prediction: {'Malignant' if pred==1 else 'Benign'} (thr={thr:.3f})")
                st.info(f"PPV: {ppv:.3f} | NPV: {npv:.3f} | Brier score: {brier:.3f}")
            else:
                st.error(f"Selected model {model_choice} does not support probability predictions.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Interactive prediction unavailable. Ensure test data and models are loaded.")

# -------------------------
# Decision Curve Analysis
# -------------------------
st.header("Decision Curve Analysis")

dca_colors = {"LR": "#1f77b4", "GB": "#ff7f0e"}
dca_frames = []
fig_dca, ax = plt.subplots()

if models and X_test is not None and y_test is not None:
    for name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                # Net benefit
                n = len(y_test)
                thr_grid = np.linspace(0.01, 0.99, 50)
                rows = []
                for thr in thr_grid:
                    y_pred = (y_prob >= thr).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    net_benefit = (tp/n) - (fp/n) * (thr / (1 - thr))
                    rows.append({"threshold": thr, "net_benefit": net_benefit})
                dca_df = pd.DataFrame(rows)
                dca_df["Model"] = name
                dca_frames.append(dca_df)

                ax.plot(dca_df["threshold"], dca_df["net_benefit"], label=name,
                        color=dca_colors.get(name, None), linewidth=2)
        except Exception as e:
            st.warning(f"Skipping DCA for {name}: {e}")

    # Baselines
    prevalence = y_test.mean()
    thr_grid = np.linspace(0.01, 0.99, 50)
    treat_all = [prevalence - (1 - prevalence) * (thr / (1 - thr)) for thr in thr_grid]
    treat_none = [0 for _ in thr_grid]
    ax.plot(thr_grid, treat_all, linestyle="--", color="black", label="Treat All")
    ax.plot(thr_grid, treat_none, linestyle=":", color="gray", label="Treat None")

    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis")
    ax.legend(loc="best")
    st.pyplot(fig_dca)
else:
    st.info("Decision Curve Analysis unavailable. Ensure test data and models are loaded.")    

# -------------------------
# Export DCA results
# -------------------------
if dca_frames:
    dca_all = pd.concat(dca_frames, ignore_index=True)
    st.subheader("Decision Curve Data")
    st.dataframe(dca_all.head(20), use_container_width=True)
    csv_data = dca_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download DCA results (CSV)",
        data=csv_data,
        file_name="decision_curve.csv",
        mime="text/csv"
    )
else:
    st.info("No DCA results available.")


 # -------------------------
# Upload Your Own Dataset
# -------------------------
st.markdown("""
### Upload Your Own Dataset
You can upload a CSV file with the same feature structure used during training.  
- The file should include all required feature columns (e.g., tumor measurements).  
- If a `diagnosis` column is present, it will be used as the label for evaluation.  
- If no label column is present, the dashboard will still generate predictions but not evaluation metrics.  
""")

try:
    expected_features = joblib.load(f"{file_path}/feature_names.pkl")
except Exception:
    expected_features = list(X_test.columns)

uploaded_file = st.file_uploader("Upload a dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    st.write("Uploaded dataset shape:", new_data.shape)

    # Check feature compatibility
    missing_features = [f for f in expected_features if f not in new_data.columns]
    if missing_features:
        st.error(f"Dataset is missing required features: {missing_features}")
    else:
        st.success("Dataset matches expected feature schema.")

        if "diagnosis" in new_data.columns:
            X_new = new_data.drop(columns=["diagnosis"])
            y_new = new_data["diagnosis"]
            st.info("Diagnosis column detected — evaluation metrics will be computed.")
            # TODO: Run evaluation (accuracy, ROC, calibration, DCA) on X_new, y_new
        else:
            X_new = new_data
            y_new = None
            st.warning("No diagnosis column found — only predictions will be available.")
            # TODO: Run predictions only

# -------------------------
# View Preprocessed Dataset
# -------------------------
st.markdown("""
### View Preprocessed Dataset
Below is the structure of the preprocessed dataset used for training and evaluation.
""")

try:
    pre_df = pd.read_csv(r"C:\Users\yasmine\Documents\Portfolio\DataSciencePortfolio\Projects\Breast-Cancer\data\preprocessed\breast_cancer_pruned.csv")
    st.write("Preprocessed dataset shape:", pre_df.shape)
    st.dataframe(pre_df.head(10))  # show first 10 rows
except Exception as e:
    st.error(f"Could not load preprocessed dataset: {e}")   

# -------------------------
# Artifacts
# -------------------------
st.subheader("Artifacts")
st.info("Figures and results are displayed directly in the dashboard. Use the download buttons above to export data.")
