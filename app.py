"""
🛒 Customer Purchase Prediction - Streamlit Web Application
E-Commerce ML Project based faithfully on Customer_Purchase_Prediction.ipynb
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI design and metric cards
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-best {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .badge-info {
        background-color: #E0F2FE;
        color: #075985;
    }
    .prediction-card-success {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .prediction-card-warning {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 2px solid #F59E0B;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .pred-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .pred-prob {
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Cached Model Loader & Pipeline Builder
# -----------------------------------------------------------------------------
@st.cache_resource
def load_or_train_pipeline():
    """
    Loads saved model package from model.pkl if available,
    otherwise executes the exact training pipeline from the notebook.
    This guarantees 100% cloud deployment compatibility.
    """
    model_file = "model.pkl"
    data_file = "customer_purchase_data.csv"

    if os.path.exists(model_file):
        try:
            return joblib.load(model_file)
        except Exception:
            pass  # Fallback to in-memory training

    if not os.path.exists(data_file):
        st.error(f"Error: Dataset '{data_file}' not found. Please ensure it is present in the repository.")
        st.stop()

    # In-memory training matching notebook exactly
    df = pd.read_csv(data_file)
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop("PurchaseStatus", axis=1)
    y = df["PurchaseStatus"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 1. Logistic Regression
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    logistic_model.fit(X_train, y_train)
    y_pred_lr = logistic_model.predict(X_test)

    # 2. Decision Tree
    decision_tree_model = DecisionTreeClassifier(
        criterion="gini", max_depth=5, random_state=42
    )
    decision_tree_model.fit(X_train, y_train)
    y_pred_dt = decision_tree_model.predict(X_test)

    # 3. KNN
    knn_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    # 4. Random Forest
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    random_forest_model.fit(X_train, y_train)
    y_pred_rf = random_forest_model.predict(X_test)

    def calc_metrics(y_true, y_pred):
        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_true, y_pred, zero_division=0))
        }

    lr_metrics = calc_metrics(y_test, y_pred_lr)
    dt_metrics = calc_metrics(y_test, y_pred_dt)
    knn_metrics = calc_metrics(y_test, y_pred_knn)
    rf_metrics = calc_metrics(y_test, y_pred_rf)

    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"],
        "Accuracy": [lr_metrics["Accuracy"], knn_metrics["Accuracy"], dt_metrics["Accuracy"], rf_metrics["Accuracy"]],
        "Precision": [lr_metrics["Precision"], knn_metrics["Precision"], dt_metrics["Precision"], rf_metrics["Precision"]],
        "Recall": [lr_metrics["Recall"], knn_metrics["Recall"], dt_metrics["Recall"], rf_metrics["Recall"]],
        "F1-Score": [lr_metrics["F1-Score"], knn_metrics["F1-Score"], dt_metrics["F1-Score"], rf_metrics["F1-Score"]]
    })

    comparison = comparison.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

    models_dict = {
        "Logistic Regression": logistic_model,
        "KNN": knn_model,
        "Decision Tree": decision_tree_model,
        "Random Forest": random_forest_model
    }

    best_model_name = comparison.loc[0, "Model"]
    best_model = models_dict[best_model_name]

    confusion_matrices = {
        "Logistic Regression": confusion_matrix(y_test, y_pred_lr).tolist(),
        "KNN": confusion_matrix(y_test, y_pred_knn).tolist(),
        "Decision Tree": confusion_matrix(y_test, y_pred_dt).tolist(),
        "Random Forest": confusion_matrix(y_test, y_pred_rf).tolist()
    }

    rf_feature_importance = pd.Series(
        random_forest_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False).to_dict()

    dt_feature_importance = pd.Series(
        decision_tree_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False).to_dict()

    artifact_package = {
        "models": models_dict,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "comparison": comparison,
        "confusion_matrices": confusion_matrices,
        "rf_feature_importance": rf_feature_importance,
        "dt_feature_importance": dt_feature_importance,
        "feature_names": feature_names,
        "metrics": {
            "Logistic Regression": lr_metrics,
            "KNN": knn_metrics,
            "Decision Tree": dt_metrics,
            "Random Forest": rf_metrics
        }
    }
    return artifact_package


# Load artifact package
package = load_or_train_pipeline()
models_dict = package["models"]
best_model_name = package["best_model_name"]
comparison_df = package["comparison"]
feature_names = package["feature_names"]
confusion_matrices = package["confusion_matrices"]
rf_feature_importance = package["rf_feature_importance"]


# -----------------------------------------------------------------------------
# 3. Sidebar — Project Information
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/shopping-cart-loaded.png", width=70)
    st.markdown("## 📌 Project Information")
    
    st.markdown("""
    **Problem:**  
    Customer Purchase Prediction  

    **Type:**  
    Binary Classification  

    **Target Variable:**  
    `PurchaseStatus`  
    - `0` = No Purchase  
    - `1` = Purchase  

    **Algorithms Evaluated:**  
    1. Logistic Regression  
    2. K-Nearest Neighbors (KNN)  
    3. Decision Tree  
    4. Random Forest  
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Model Selection")
    
    # Model selector (defaulting to the best model dynamically)
    model_options = list(models_dict.keys())
    best_idx = model_options.index(best_model_name) if best_model_name in model_options else 0
    
    selected_model_name = st.selectbox(
        "Choose Model for Prediction:",
        options=model_options,
        index=best_idx,
        help="Defaults to the best performing model based on F1-Score."
    )
    
    selected_model = models_dict[selected_model_name]
    
    if selected_model_name == best_model_name:
        st.success(f"🏆 Best Model Selected: **{best_model_name}**")
    else:
        st.info(f"Selected: **{selected_model_name}** (Best: {best_model_name})")

    st.markdown("---")
    st.caption("College ML Project | E-Commerce Analytics")


# -----------------------------------------------------------------------------
# 4. Main Page Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🛒 Customer Purchase Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Predict whether an e-commerce customer is likely to make a purchase based on their profile and shopping behavior.</div>',
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 5. Customer Profile & Shopping Behavior Inputs
# -----------------------------------------------------------------------------
st.markdown("### 👤 Enter Customer Details & Shopping Behavior")

# Organized in 2 columns for a balanced layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 📋 Customer Profile")
    
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
        help="Customer age in years (dataset range: 18 - 70)"
    )
    
    gender_label = st.radio(
        "Gender",
        options=["Female (0)", "Male (1)"],
        index=1,
        horizontal=True,
        help="Customer gender (0 = Female, 1 = Male)"
    )
    gender = 1 if "1" in gender_label else 0

    annual_income = st.number_input(
        "Annual Income ($)",
        min_value=10000.0,
        max_value=300000.0,
        value=80000.0,
        step=1000.0,
        format="%.2f",
        help="Customer annual income in USD"
    )

    loyalty_label = st.radio(
        "Loyalty Program Member",
        options=["No (0)", "Yes (1)"],
        index=1,
        horizontal=True,
        help="Whether the customer belongs to the store loyalty program"
    )
    loyalty_program = 1 if "1" in loyalty_label else 0

with col2:
    st.markdown("##### 🛍️ Website & Shopping Behavior")
    
    number_of_purchases = st.number_input(
        "Number of Previous Purchases",
        min_value=0,
        max_value=100,
        value=10,
        step=1,
        help="Total previous purchases made by the customer"
    )

    product_cat_label = st.selectbox(
        "Product Category Browsed",
        options=[
            "Category 0 (Electronics)",
            "Category 1 (Clothing)",
            "Category 2 (Home & Kitchen)",
            "Category 3 (Beauty & Health)",
            "Category 4 (Books & Media)"
        ],
        index=2,
        help="Numeric encoded product category (0 to 4)"
    )
    product_category = int(product_cat_label.split()[1])

    time_spent = st.number_input(
        "Time Spent on Website (minutes)",
        min_value=0.5,
        max_value=180.0,
        value=35.0,
        step=0.5,
        format="%.2f",
        help="Total browsing session duration in minutes"
    )

    discounts_availed = st.number_input(
        "Discounts Availed",
        min_value=0,
        max_value=20,
        value=3,
        step=1,
        help="Number of discount coupons or offers used"
    )

# Construct the input DataFrame matching the EXACT feature names and order
input_data = {
    "Age": [age],
    "Gender": [gender],
    "AnnualIncome": [annual_income],
    "NumberOfPurchases": [number_of_purchases],
    "ProductCategory": [product_category],
    "TimeSpentOnWebsite": [time_spent],
    "LoyaltyProgram": [loyalty_program],
    "DiscountsAvailed": [discounts_availed]
}

input_df = pd.DataFrame(input_data)[feature_names]


# -----------------------------------------------------------------------------
# 6. Prediction Action & Dynamic Results
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🚀 Predict Purchase", type="primary", use_container_width=True)

if predict_btn:
    # Generate prediction and exact probability using predict_proba()
    prediction = int(selected_model.predict(input_df)[0])
    probabilities = selected_model.predict_proba(input_df)[0]
    purchase_prob = probabilities[1]
    no_purchase_prob = probabilities[0]

    st.markdown("---")
    st.markdown("### 🎯 Prediction Results")
    
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-card-success">
                <div class="pred-title" style="color: #065F46;">✅ Customer Will Purchase</div>
                <div class="pred-prob" style="color: #047857;">Purchase Probability: {purchase_prob:.2%}</div>
                <p style="margin-top: 10px; color: #064E3B; font-size: 0.95rem;">
                    The model predicts a high likelihood of purchase conversion for this customer profile.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card-warning">
                <div class="pred-title" style="color: #92400E;">⚠️ Customer Will Not Purchase</div>
                <div class="pred-prob" style="color: #B45309;">Purchase Probability: {purchase_prob:.2%}</div>
                <p style="margin-top: 10px; color: #78350F; font-size: 0.95rem;">
                    The customer is currently unlikely to complete a purchase. Consider offering targeted incentives.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("##### 📈 Probability Breakdown")
        st.write(f"**Likelihood of Purchase (Class 1):** `{purchase_prob:.2%}`")
        st.progress(float(purchase_prob))
        
        st.write(f"**Likelihood of No Purchase (Class 0):** `{no_purchase_prob:.2%}`")
        st.progress(float(no_purchase_prob))

        st.caption(f"Evaluated using: **{selected_model_name}**")


# -----------------------------------------------------------------------------
# 7. Model Performance Section
# -----------------------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("## 📊 Model Performance")

st.markdown(f"""
All models are trained and evaluated on an **80:20 stratified train-test split** using the exact parameters from the Jupyter Notebook.
The **Best Model** is chosen dynamically based on highest **F1-Score**.
""")

# Best model highlight banner
st.success(f"""
🏆 **Best Performing Model: {best_model_name}**  
Achieved the highest F1-Score of **{comparison_df.loc[0, 'F1-Score']:.4f}** and Accuracy of **{comparison_df.loc[0, 'Accuracy']:.4f}**.
""")

perf_tab1, perf_tab2 = st.tabs(["📋 Performance Metrics Table", "📊 Model Comparison Chart"])

with perf_tab1:
    # Formatted comparison table
    formatted_comparison = comparison_df.copy()
    formatted_comparison["Accuracy"] = formatted_comparison["Accuracy"].apply(lambda x: f"{x:.4f}")
    formatted_comparison["Precision"] = formatted_comparison["Precision"].apply(lambda x: f"{x:.4f}")
    formatted_comparison["Recall"] = formatted_comparison["Recall"].apply(lambda x: f"{x:.4f}")
    formatted_comparison["F1-Score"] = formatted_comparison["F1-Score"].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(formatted_comparison, use_container_width=True, hide_index=True)

with perf_tab2:
    # Model comparison bar chart matching notebook aesthetic
    fig, ax = plt.subplots(figsize=(10, 4.5))
    metrics_plot_df = comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]]
    
    metrics_plot_df.plot(
        kind="bar",
        ax=ax,
        colormap="Blues",
        edgecolor="#1E293B",
        linewidth=0.8,
        width=0.75
    )
    ax.set_title("Model Performance Comparison across Key Metrics", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#E2E8F0")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# -----------------------------------------------------------------------------
# 8. Feature Importance & Confusion Matrix
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    st.markdown("### 🌲 Feature Importance")
    st.markdown("Top features influencing predictions in **Random Forest**:")
    
    feat_df = pd.DataFrame(
        list(rf_feature_importance.items()),
        columns=["Feature", "Importance"]
    ).sort_values(by="Importance", ascending=True)

    fig_imp, ax_imp = plt.subplots(figsize=(7, 4.8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_df)))
    bars = ax_imp.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor="black", linewidth=0.5)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax_imp.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}",
                    ha="left", va="center", fontsize=8.5, color="#1E293B")
        
    ax_imp.set_title("Random Forest - Feature Importance", fontsize=11, fontweight="bold", pad=10)
    ax_imp.set_xlabel("Importance Score", fontsize=9.5)
    ax_imp.set_xlim(0, max(feat_df["Importance"]) * 1.2)
    ax_imp.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_imp)
    plt.close()
    
    st.caption("💡 **Key Insight:** `TimeSpentOnWebsite`, `Age`, and `AnnualIncome` are the most influential purchase predictors.")

with vis_col2:
    st.markdown(f"### 🔲 Confusion Matrix")
    st.markdown(f"Test Set Confusion Matrix for **{selected_model_name}**:")
    
    cm = np.array(confusion_matrices[selected_model_name])
    
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Purchase (0)", "Purchase (1)"],
        yticklabels=["No Purchase (0)", "Purchase (1)"],
        ax=ax_cm,
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax_cm.set_title(f"{selected_model_name} - Confusion Matrix", fontsize=11, fontweight="bold", pad=10)
    ax_cm.set_xlabel("Predicted Label", fontsize=9.5, fontweight="bold")
    ax_cm.set_ylabel("True Label", fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close()

    total_test = cm.sum()
    correct_test = cm[0, 0] + cm[1, 1]
    st.caption(f"✅ Correctly classified **{correct_test}/{total_test}** test instances ({correct_test/total_test:.2%} test accuracy).")


# -----------------------------------------------------------------------------
# 9. Verification & Reference Sample Data
# -----------------------------------------------------------------------------
with st.expander("🔍 Test Sample from Notebook (Verification Data)"):
    st.markdown("""
    You can test the exact verification sample from the Jupyter Notebook:
    - **Age**: `30`
    - **Gender**: `Male (1)`
    - **Annual Income**: `$80,000`
    - **Number of Purchases**: `10`
    - **Product Category**: `Category 2`
    - **Time Spent on Website**: `35.0 minutes`
    - **Loyalty Program**: `Yes (1)`
    - **Discounts Availed**: `3`
    
    **Expected Outcome:** `Customer Will Purchase` with high probability (~97%+ on Random Forest).
    """)

st.markdown("---")
st.markdown(
    "<center><small style='color: #64748B;'>Customer Purchase Prediction App • Built with Streamlit, Scikit-Learn & Pandas</small></center>",
    unsafe_allow_html=True
)
