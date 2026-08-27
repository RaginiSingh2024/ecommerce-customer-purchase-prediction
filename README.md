# 🛒 Customer Purchase Prediction from E-Commerce

A machine learning web application that predicts whether an e-commerce customer is likely to make a purchase based on their demographic profile and website browsing behavior.

Built with **Python**, **Scikit-Learn**, **Pandas**, and **Streamlit**.

---

## 📌 Project Overview

- **Problem Type:** Binary Classification
- **Target Variable:** `PurchaseStatus`
  - `0` = No Purchase
  - `1` = Purchase
- **Algorithms Evaluated:**
  1. **Logistic Regression** (with `StandardScaler` pipeline)
  2. **K-Nearest Neighbors (KNN)** (with `StandardScaler` pipeline, $k=5$)
  3. **Decision Tree** (`criterion='gini'`, `max_depth=5`)
  4. **Random Forest** (`n_estimators=100`, `max_depth=10`)
- **Model Selection Criteria:** Selected dynamically based on the highest **F1-Score** (Random Forest achieves **~0.9225 F1-Score** and **~92.81% Test Accuracy**).

---

## 📁 Project Structure

```text
customer-purchase-prediction/
│
├── app.py                            # Main Streamlit web application
├── train_model.py                    # Standalone model training & evaluation script
├── requirements.txt                  # Required Python dependencies
├── customer_purchase_data.csv        # Dataset (1,388 cleaned records)
├── model.pkl                         # Serialized models and performance metrics
├── Customer_Purchase_Prediction.ipynb# Original Jupyter Notebook
└── README.md                         # Project documentation and deployment guide
```

---

## 🚀 Running the App Locally

### 1. Prerequisites
Make sure you have Python 3.9+ installed.

### 2. Clone or Navigate to the Directory
```bash
cd customer-purchase-prediction
```

### 3. (Optional) Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate    # On macOS/Linux
# or: venv\Scripts\activate # On Windows
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. (Optional) Re-train and Generate `model.pkl`
```bash
python train_model.py
```

### 6. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser and visit `http://localhost:8501`.

---

## 🌐 Step-by-Step GitHub Upload Instructions

### Option A: Using Git Command Line (Recommended)

1. Open your terminal and navigate to the project directory:
   ```bash
   cd customer-purchase-prediction
   ```

2. Initialize a git repository:
   ```bash
   git init
   ```

3. Add all project files:
   ```bash
   git add .
   ```

4. Commit the changes:
   ```bash
   git commit -m "Initial commit: Customer Purchase Prediction Streamlit App"
   ```

5. Create a new repository on [GitHub](https://github.com/new) named `customer-purchase-prediction` (set it to **Public**).

6. Link your local repo and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/customer-purchase-prediction.git
   git push -u origin main
   ```

---

### Option B: Uploading via GitHub Website (No Git CLI needed)

1. Go to [GitHub](https://github.com) and click **"New repository"**.
2. Name it `customer-purchase-prediction` and click **"Create repository"**.
3. On the setup page, click the link: **"uploading an existing file"**.
4. Drag and drop the following files into the upload box:
   - `app.py`
   - `requirements.txt`
   - `customer_purchase_data.csv`
   - `train_model.py`
   - `model.pkl`
   - `Customer_Purchase_Prediction.ipynb`
   - `README.md`
5. Click **"Commit changes"**.

---

## ☁️ Step-by-Step Deployment on Streamlit Community Cloud

Deploying your project to Streamlit Community Cloud is **100% free**:

1. **Sign in to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.

2. **Create New App:**
   - Click the blue **"New app"** button.

3. **Configure App Settings:**
   - **Repository:** Select your GitHub repository (`<YOUR_USERNAME>/customer-purchase-prediction`).
   - **Branch:** `main`
   - **Main file path:** `app.py`

4. **Deploy:**
   - Click **"Deploy!"**.
   - Streamlit Cloud will automatically install dependencies from `requirements.txt` and launch your live web application in seconds.

---

## 📋 Exact Files Required for GitHub / Cloud Deployment

| File | Purpose | Required for Cloud? |
| :--- | :--- | :--- |
| `app.py` | Streamlit user interface & prediction logic | **Yes (Mandatory)** |
| `requirements.txt` | Dependency list for Streamlit Cloud builder | **Yes (Mandatory)** |
| `customer_purchase_data.csv` | Dataset for caching and training fallback | **Yes (Mandatory)** |
| `model.pkl` | Pre-trained model & metric cache for instant loading | **Yes (Recommended)** |
| `train_model.py` | Standalone script for training & evaluation | Optional / Recommended |
| `Customer_Purchase_Prediction.ipynb` | Project reference notebook | Optional / Recommended |
| `README.md` | Documentation | Optional / Recommended |

---

## 📊 Evaluation Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Best)** 🏆 | **92.81%** | **92.97%** | **91.54%** | **0.9225** |
| K-Nearest Neighbors (KNN) | 85.25% | 81.12% | 89.23% | 0.8498 |
| Decision Tree | 85.25% | 82.01% | 87.69% | 0.8476 |
| Logistic Regression | 83.09% | 81.68% | 82.31% | 0.8199 |

---

## 🧪 Verification Sample

Use these inputs to verify the app predictions against the notebook:
- **Age:** `30`
- **Gender:** `Male (1)`
- **Annual Income:** `80000`
- **Number of Purchases:** `10`
- **Product Category:** `Category 2`
- **Time Spent on Website:** `35.0`
- **Loyalty Program:** `Yes (1)`
- **Discounts Availed:** `3`

**Result:** `Customer Will Purchase` (Purchase Probability: `~97.28%`).
