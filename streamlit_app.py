"""
Digit Lab: interactive evaluation of 5 pre-trained classifiers on handwritten digits.
Loads saved models from model/*.pkl (no retraining) and scores an uploaded test CSV
that contains the 64 pixel columns + the `digit` target column.
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report, roc_curve, auc)

st.set_page_config(page_title="Handwritten Digits Classifier", page_icon="🔢", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.markdown("""
<style>
.block-container {padding-top: 1.4rem;}
.hero {background: linear-gradient(120deg,#7c3aed,#0ea5e9); color:white; padding:1.2rem 1.6rem;
       border-radius:14px; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:1.8rem;} .hero p {margin:.2rem 0 0; opacity:.92;}
div[data-testid="stMetric"] {background:#f5f3ff; border-radius:12px; padding:.7rem 1rem; border:1px solid #ddd6fe;}
div[data-testid="stMetric"] label, div[data-testid="stMetricValue"] {color:#1e1b4b !important;}
</style>
<div class="hero"><h1>🔢 Handwritten Digits Classifier</h1>
<p>Explore five trained classifiers on 8×8 handwritten digits on UCI Optical Recognition dataset. Upload a test CSV and compare them.</p></div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_FILES[name])


@st.cache_data
def load_meta():
    return json.load(open("model/meta.json"))


def compute(model, X, y, classes):
    pred, proba = model.predict(X), model.predict_proba(X)
    return pred, proba, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="weighted", labels=classes),
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred)}


meta = load_meta()
target, features, classes = meta["target"], meta["features"], meta["classes"]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Options")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))
    up = st.file_uploader("Upload test CSV", type="csv",
                          help=f"Needs the {len(features)} pixel columns and the '{target}' column.")
    use_sample = st.checkbox("Use bundled test_data.csv", value=up is None, disabled=up is not None)
    st.caption(f"Target: **{target}** · Features: **{len(features)}** · Classes: **{len(classes)}**")
    with open("test_data.csv", "rb") as f:
        st.download_button("⬇️ Download sample test CSV", f, "test_data.csv", "text/csv")
    st.divider()
    st.caption(f"Dataset: {meta['dataset']}  \n"
               f"Models trained once in `train_models.ipynb` (scikit-learn {meta['sklearn_version']}) "
               "and loaded from `model/` — nothing is retrained here.")

if up is not None:
    df = pd.read_csv(up)
elif use_sample:
    df = pd.read_csv("test_data.csv")
else:
    st.info("Upload a CSV or tick the bundled sample in the sidebar to begin.")
    st.stop()

missing = [c for c in features + [target] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    st.stop()
if not set(df[target].unique()) <= set(classes):
    st.error(f"Column '{target}' contains labels outside {classes}.")
    st.stop()

X, y = df[features], df[target]
model = load_model(model_name)
pred, proba, m = compute(model, X, y, classes)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Single model", "🏁 Compare all models", "🖼️ Try a digit", "🔎 Data preview"])

# ---------------- Tab 1 ----------------
with tab1:
    st.subheader(f"{model_name}: results on {len(df)} rows")
    for c, k in zip(st.columns(6), METRICS):
        c.metric(k, f"{m[k]:.4f}")
    st.caption("Precision, Recall, F1 and AUC are weighted averages over the 10 digit classes (AUC is one-vs-rest).")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(5, 4.2))
        sns.heatmap(confusion_matrix(y, pred, labels=classes), annot=True, fmt="d", cmap="Purples",
                    cbar=False, ax=ax, xticklabels=classes, yticklabels=classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**ROC curves (one-vs-rest, per digit)**")
        Y = label_binarize(y, classes=classes)
        fig2, ax2 = plt.subplots(figsize=(5, 4.2))
        for i, cl in enumerate(classes):
            fpr, tpr, _ = roc_curve(Y[:, i], proba[:, i])
            ax2.plot(fpr, tpr, lw=1.2, label=f"{cl} (AUC {auc(fpr, tpr):.2f})")
        ax2.plot([0, 1], [0, 1], "k--", lw=1)
        ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate")
        ax2.legend(fontsize=6, ncol=2, loc="lower right")
        st.pyplot(fig2)

    st.markdown("**Classification report**")
    st.dataframe(pd.DataFrame(classification_report(y, pred, output_dict=True, zero_division=0)).T.round(3),
                 use_container_width=True)

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("All five models on the same data")
    comp = pd.DataFrame({n: compute(load_model(n), X, y, classes)[2] for n in MODEL_FILES}).T[METRICS]
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#ddd6fe"), use_container_width=True)
    pick = st.radio("Plot metric", METRICS, horizontal=True)
    st.bar_chart(comp[pick])
    best = comp[pick].idxmax()
    st.success(f"Best by **{pick}**: **{best}** ({comp.loc[best, pick]:.4f})")
    st.download_button("⬇️ Download comparison CSV", comp.round(4).to_csv().encode(),
                       "model_comparison.csv", "text/csv")

# ---------------- Tab 3 ----------------
with tab3:
    st.subheader(f"Pick a row and see what {model_name} predicts")
    only_wrong = st.checkbox("Show only misclassified rows", value=False)
    idx_pool = np.where(pred != y.values)[0] if only_wrong else np.arange(len(df))
    if len(idx_pool) == 0:
        st.success("No misclassified rows for this model on this data.")
    else:
        pos = st.slider("Row", 0, len(idx_pool) - 1, 0) if len(idx_pool) > 1 else 0
        i = int(idx_pool[pos])
        c1, c2 = st.columns([1, 2])
        with c1:
            fig3, ax3 = plt.subplots(figsize=(3, 3))
            ax3.imshow(X.iloc[i].values.reshape(8, 8), cmap="gray_r"); ax3.axis("off")
            st.pyplot(fig3)
            ok = pred[i] == y.iloc[i]
            st.markdown(f"**Actual:** {y.iloc[i]}  \n**Predicted:** {pred[i]}  {'✅' if ok else '❌'}")
        with c2:
            st.markdown("**Predicted probability per digit**")
            st.bar_chart(pd.Series(proba[i], index=[str(c) for c in classes]))

# ---------------- Tab 4 ----------------
with tab4:
    st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(50), use_container_width=True)
    st.write("Class distribution")
    st.bar_chart(y.value_counts().sort_index())
