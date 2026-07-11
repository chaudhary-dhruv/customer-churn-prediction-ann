# 📉 Customer Churn Prediction System using Artificial Neural Network (ANN)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ANN-FF6F00?style=for-the-badge\&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge\&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-F7931E?style=for-the-badge\&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### Predict customer churn using an Artificial Neural Network and an interactive Streamlit web application.

🔗 **Live Demo:** https://customer-churn-prediction-ann-mmbqdbqilq4hggzc3abheo.streamlit.app/

</div>

---

# 📌 Project Overview

Customer churn is one of the biggest business challenges in industries like banking, telecom, insurance, and subscription-based services. Losing existing customers directly impacts revenue and profitability.

This project uses an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras** to predict whether a customer is likely to leave the bank based on demographic and financial information.

The trained model is deployed using **Streamlit Cloud**, allowing users to interactively enter customer details and receive real-time churn predictions.

---

# 🚀 Live Demo

### 🌐 Streamlit Application

https://customer-churn-prediction-ann-mmbqdbqilq4hggzc3abheo.streamlit.app/

---

# ✨ Features

* 🤖 Artificial Neural Network (ANN) for binary classification
* 📊 Real-time customer churn prediction
* 🌍 Geography One-Hot Encoding
* 👤 Gender Label Encoding
* 📈 Feature Standardization using StandardScaler
* ⚡ Interactive Streamlit interface
* ☁️ Cloud deployment using Streamlit Cloud
* 💾 Saved preprocessing pipeline using Pickle
* 🔄 End-to-end inference pipeline

---

# 🧠 Model Workflow

```text
Customer Details
        │
        ▼
Data Preprocessing
(Label Encoding + One-Hot Encoding)
        │
        ▼
Feature Scaling
(StandardScaler)
        │
        ▼
Artificial Neural Network
(TensorFlow/Keras)
        │
        ▼
Churn Probability
        │
        ▼
Likely to Churn / Not Likely to Churn
```

---

# 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── experiments.ipynb
├── prediction.ipynb
├── model.h5
├── scaler.pkl
├── label_encoder_gender.pkl
├── one_hot_encoder_geo.pkl
├── requirements.txt
├── runtime.txt
├── README.md
│
└── dataset/
```

---

# 🛠️ Tech Stack

| Category             | Technology         |
| -------------------- | ------------------ |
| Programming Language | Python 3.11        |
| Machine Learning     | TensorFlow / Keras |
| Data Processing      | Pandas, NumPy      |
| Preprocessing        | Scikit-Learn       |
| Web Framework        | Streamlit          |
| Model Deployment     | Streamlit Cloud    |
| Version Control      | Git & GitHub       |

---

# 📊 Input Features

The model uses the following customer attributes:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary

---

# 📈 Model Output

The application predicts:

* ✅ Customer is **Not Likely to Churn**
* ⚠️ Customer is **Likely to Churn**

along with the predicted probability.

Example:

```text
Customer is likely to churn.

Probability: 93.72%
```

---

# ⚙️ Installation

Clone the repository

```bash
https://github.com/chaudhary-dhruv/customer-churn-prediction-ann
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---



---

# 🎯 Future Improvements

* SHAP Explainability
* Prediction History
* Download Prediction Report
* Docker Containerization
* CI/CD Pipeline using GitHub Actions
* Cloud Deployment using AWS or Azure
* User Authentication
* Database Integration

---

# 📚 Dataset

**Kaggle Customer Churn Dataset**

The dataset contains customer demographic, financial, and account information used to predict customer churn.

---

# 👨‍💻 Author

**Dhruv Chaudhary**

GitHub:
https://github.com/chaudhary-dhruv

LinkedIn:
https://www.linkedin.com/in/dhruv-chaudhary-8026a3258/

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub. It helps increase visibility and motivates further development.

---
