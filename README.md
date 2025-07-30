# 🧠 Stroke Prediction Using Machine Learning

This project aims to predict whether a patient is likely to have a stroke based on various health indicators. It uses supervised machine learning algorithms and involves data cleaning, feature selection, and model evaluation.

## 📌 Project Overview

Stroke is one of the leading causes of death and disability worldwide. Early detection can help save lives. This project uses a real-world dataset to build a classification model that predicts stroke risk with over 85% accuracy.

## 🔧 Tools & Technologies

- **Python**
- **Pandas, NumPy** – Data handling & preprocessing
- **Matplotlib, Seaborn** – Data visualization
- **Scikit-learn** – Model building & evaluation
- **Jupyter Notebook** – Development environment

## 📊 Dataset

- The dataset includes features such as age, gender, hypertension, heart disease, work type, smoking status, BMI, and average glucose level.
- Target variable: `stroke` (0 = No stroke, 1 = Stroke)

_Source: Kaggle

## 🔍 Process

1. **Data Cleaning**
   - Handled missing values and inconsistent data
   - Encoded categorical variables

2. **Exploratory Data Analysis (EDA)**
   - Visualized feature distributions and correlations
   - Identified key risk indicators

3. **Feature Selection**
   - Selected relevant features to improve model performance

4. **Modeling**
   - Trained multiple models:  
     - Logistic Regression  
     - K-Nearest Neighbors (KNN)  
     - Decision Tree  
     - Random Forest
   - Compared models based on accuracy, precision, recall, F1-score

5. **Evaluation**
   - Used confusion matrix and classification report
   - Final model accuracy: **~85%**

## 📈 Results

- **Random Forest** performed best with high accuracy and balanced precision/recall.
- Visualizations helped understand data distribution and key influencing factors.


## 📌 Key Learnings

- Importance of data preprocessing and feature selection
- Model evaluation beyond just accuracy
- Applying real-world health data in meaningful ML use cases

## ✅ Future Improvements

- Deploy the model using Streamlit or Flask
- Integrate with a real-time health monitoring dashboard
- Apply hyperparameter tuning (GridSearchCV)

## 📬 Contact

**Dylan Dias**  
📧 dylandias73@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/dylan-dias-7937492a4)

---



