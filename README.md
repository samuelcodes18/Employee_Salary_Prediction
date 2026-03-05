# 💼 Employee Salary Prediction  
*A Machine Learning-based project developed as part of the IBM SkillsBuild Internship*

---

## 📖 Project Overview

This project aims to build a **binary classification** machine learning model that predicts whether an individual's annual salary exceeds **$50,000** based on various demographic and professional features.

Key steps in this project include:
- Data loading and cleaning  
- Exploratory Data Analysis (EDA)  
- Model training and evaluation  
- Deployment using **Streamlit** for interactive usage  

The dataset used is a cleaned and modified version of the **UCI Adult Income** dataset.

---

## 📊 Dataset

The dataset used is `Salary_List.csv`, containing the following features:

| Feature           | Description |
|-------------------|-------------|
| `age`             | Age of the individual |
| `workclass`       | Type of employer (e.g., Private, Federal-gov) |
| `fnlwgt`          | Final weight (sampling weight) |
| `education`       | Education level (e.g., Bachelors, HS-grad) |
| `education-num`   | Numeric representation of education |
| `marital-status`  | Marital status |
| `occupation`      | Occupation type |
| `relationship`    | Relationship status |
| `race`            | Race |
| `sex`             | Gender |
| `capital-gain`    | Capital gains |
| `capital-loss`    | Capital losses |
| `hours-per-week`  | Hours worked per week |
| `native-country`  | Country of origin |
| `salary`          | Target label (<=50K or >50K) |

### 🧹 Data Preprocessing

- Handled missing values (`?`)  
- Removed irrelevant entries (`Without-pay`, `Never-worked`)  
- Dropped redundant column (`education`, since `education-num` provides numeric representation)  
- Treated outliers in `age` and `education-num`  
- Encoded categorical features using **Label Encoding**  

---

## 📁 Project Structure
Employee_Salary_Prediction/

│

├── app.py # Streamlit app

├── model.joblib # Saved ML model

├── Salary_List.csv # Dataset

├── salary_prediction.ipynb # Jupyter notebook for development

├── requirements.txt # Required libraries

└── README.md # Project documentation


---

## ⚙️ Setup and Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/samuelcodes18/Employee_Salary_Prediction.git
cd Employee_Salary_Prediction
```
```bash
pip install -r requirements.txt
```
```bash
ngrok authtoken YOUR_NGROK_AUTHTOKEN
```
```bash
streamlit run app.py
```

