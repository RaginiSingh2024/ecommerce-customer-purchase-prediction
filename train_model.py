"""
Customer Purchase Prediction - Model Training Script
Preserves the exact preprocessing, hyperparameters, and logic from Customer_Purchase_Prediction.ipynb.
"""

import os
import joblib
import pandas as pd
import numpy as np
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

def train_and_save_models(data_path="customer_purchase_data.csv", output_path="model.pkl"):
    print("1. Loading dataset from:", data_path)
    df = pd.read_csv(data_path)
    
    # 2. Data Cleaning - remove duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"   Cleaned duplicates: {before - after} removed ({after} records remaining)")

    # 3. Feature and Target Separation
    X = df.drop("PurchaseStatus", axis=1)
    y = df["PurchaseStatus"]
    feature_names = X.columns.tolist()
    print("   Features:", feature_names)
    print("   Target: PurchaseStatus")

    # 4. Train-Test Split (Exact parameters from notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 5. Model Definitions & Training
    print("5. Training Models...")
    
    # Model 1: Logistic Regression with StandardScaler Pipeline
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    logistic_model.fit(X_train, y_train)
    y_pred_lr = logistic_model.predict(X_test)

    # Model 2: Decision Tree Classifier
    decision_tree_model = DecisionTreeClassifier(
        criterion="gini", max_depth=5, random_state=42
    )
    decision_tree_model.fit(X_train, y_train)
    y_pred_dt = decision_tree_model.predict(X_test)

    # Model 3: K-Nearest Neighbors (KNN) with StandardScaler Pipeline
    knn_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    # Model 4: Random Forest Classifier
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    random_forest_model.fit(X_train, y_train)
    y_pred_rf = random_forest_model.predict(X_test)

    # 6. Model Evaluation
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

    # Model comparison table
    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"],
        "Accuracy": [lr_metrics["Accuracy"], knn_metrics["Accuracy"], dt_metrics["Accuracy"], rf_metrics["Accuracy"]],
        "Precision": [lr_metrics["Precision"], knn_metrics["Precision"], dt_metrics["Precision"], rf_metrics["Precision"]],
        "Recall": [lr_metrics["Recall"], knn_metrics["Recall"], dt_metrics["Recall"], rf_metrics["Recall"]],
        "F1-Score": [lr_metrics["F1-Score"], knn_metrics["F1-Score"], dt_metrics["F1-Score"], rf_metrics["F1-Score"]]
    })

    # Sort dynamically by F1-score as done in notebook
    comparison = comparison.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

    models_dict = {
        "Logistic Regression": logistic_model,
        "KNN": knn_model,
        "Decision Tree": decision_tree_model,
        "Random Forest": random_forest_model
    }

    best_model_name = comparison.loc[0, "Model"]
    best_model = models_dict[best_model_name]

    # Confusion matrices
    confusion_matrices = {
        "Logistic Regression": confusion_matrix(y_test, y_pred_lr).tolist(),
        "KNN": confusion_matrix(y_test, y_pred_knn).tolist(),
        "Decision Tree": confusion_matrix(y_test, y_pred_dt).tolist(),
        "Random Forest": confusion_matrix(y_test, y_pred_rf).tolist()
    }

    # Feature importances
    rf_feature_importance = pd.Series(
        random_forest_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False).to_dict()

    dt_feature_importance = pd.Series(
        decision_tree_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False).to_dict()

    print("\n=== Model Performance Comparison ===")
    print(comparison.round(4).to_string(index=False))
    print(f"\nBest Model by F1-Score: {best_model_name} ({comparison.loc[0, 'F1-Score']:.4f})")

    # 7. Save model package
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

    joblib.dump(artifact_package, output_path)
    print(f"\nSuccessfully saved model package to '{output_path}'.")
    return artifact_package

if __name__ == "__main__":
    train_and_save_models()
