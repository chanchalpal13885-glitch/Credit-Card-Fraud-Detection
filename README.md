# 💳 Credit Card Fraud Detection System
### Machine Learning • Classification • Financial Risk Analytics

---

## 📘 Project Overview
This project presents a machine learning–based credit card fraud detection system designed to identify fraudulent transactions from a highly imbalanced financial dataset.

Credit card fraud is a major challenge in the banking and fintech industries, where undetected fraud can result in significant financial losses. This project demonstrates how supervised machine learning models can effectively learn transaction patterns and detect fraudulent behavior.

The complete workflow — including data loading, preprocessing, model training, and evaluation — is implemented using Python in a Jupyter Notebook, ensuring clarity, transparency, and reproducibility.

---

## 🎯 Business Objective
To develop and compare multiple classification models that can accurately distinguish between fraudulent and legitimate credit card transactions while handling extreme class imbalance.

---

## 🧾 Problem Statement
Fraudulent transactions account for a very small percentage of total credit card transactions. Due to this imbalance, traditional accuracy-based evaluation can be misleading.

The key challenge is to maximize fraud detection performance, particularly recall for fraudulent transactions, while minimizing false negatives that can lead to financial risk.

---

## 📊 Dataset Description
- **Dataset Name:** Credit Card Fraud Detection Dataset
- **Source:** Kaggle (Université Libre de Bruxelles)
- **Data Type:** Structured tabular data
- **Format:** CSV

### ⚠️ Dataset Availability Notice
Due to GitHub file size limitations, the dataset is not included in this repository.

### 🔗 Official Dataset Link
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🔑 Feature Overview
| Feature | Description |
|------|------------|
| Time | Seconds elapsed between transactions |
| V1 – V28 | Anonymized PCA-transformed numerical features |
| Amount | Transaction amount |
| Class | Target variable (0 = Legitimate, 1 = Fraud) |

> The dataset is highly imbalanced, with fraudulent transactions representing less than 1% of total records.

---

## 🧠 Machine Learning Models Implemented
The following supervised learning algorithms were trained and evaluated:

- **Logistic Regression** – Baseline probabilistic classifier
- **Decision Tree Classifier** – Rule-based and interpretable model
- **Random Forest Classifier** – Ensemble model for improved robustness
- **K-Nearest Neighbors (KNN)** – Distance-based classification approach
- **Linear Support Vector Machine (Linear SVM)** – Margin-based classifier suitable for high-dimensional data

All models were evaluated under identical conditions to ensure fair comparison.

---

## 📈 Evaluation Metrics
Given the imbalanced nature of the dataset, performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Special emphasis was placed on recall for the fraud class, as missing fraudulent transactions can have serious financial consequences.

---

## 🛠️ Skills Demonstrated
- Data preprocessing and analysis using Pandas
- Handling imbalanced classification problems
- Training and comparison of multiple ML models
- Model evaluation using appropriate performance metrics
- End-to-end machine learning workflow implementation
- Reproducible analysis using Jupyter Notebook

---

## 🧰 Tools & Technologies
- Python
- Jupyter Notebook (Anaconda)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## 🗂️ Repository Structure
Credit-Card-Fraud-Detection

├── Credit Card Fraud Detection.ipynb

└── README.md


---

## ▶️ How to Run the Project
1. Download the dataset from the Kaggle link provided above
2. Place `creditcard.csv` in the same directory as the notebook
3. Open the notebook using:
   - Jupyter Notebook
   - JupyterLab
   - Anaconda Navigator
4. Run all cells sequentially to reproduce the results

---

## ⚠️ Ethical & Practical Considerations
- The dataset is fully anonymized to protect user privacy
- This project is intended strictly for academic and portfolio purposes
- No deployment is included; the focus is on model development and evaluation
- Real-world implementation would require regulatory compliance and continuous monitoring

---

## 🚀 Future Enhancements
- Advanced imbalance handling techniques (SMOTE, undersampling)
- ROC-AUC and Precision–Recall curve analysis
- Feature importance and model explainability
- Hyperparameter optimization
- Model deployment using Streamlit or Flask
- Business dashboards using Power BI or Tableau

---

## 👩‍💻 Author
**Chanchal Pal**  
📊 Data Analyst | Machine Learning Enthusiast | Aspiring Data Scientist  

🔗 GitHub: https://github.com/chanchalpal13885-glitch  
🔗 LinkedIn: https://www.linkedin.com/in/chanchalpal  
📧 Email: chanchalpal13885@gmail.com

---

## ⭐ Acknowledgment
If you find this project useful, please consider giving it a ⭐ on GitHub.
