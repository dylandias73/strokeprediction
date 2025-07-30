import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import streamlit as st

# Load Dataset
df = pd.read_csv("stroke.csv")

# Rename target column to stroke
df.rename(columns={'target': 'stroke'}, inplace=True)

# Display first few rows
print(df.head())

# Handle Missing Values
df.fillna(df.mean(), inplace=True)

# Exploratory Data Analysis
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Splitting Data
X = df.drop(columns=['stroke'])  # 'stroke' is the new target variable
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save feature names
feature_names = X.columns.tolist()

# Model Training
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Save the best model and scaler
best_model = max(results, key=results.get)
joblib.dump(models[best_model], 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Deployment with Streamlit
def predict_stroke(input_data):
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return 'Stroke Detected' if prediction[0] == 1 else 'No Stroke'

st.title("Stroke Prediction App")
st.markdown(
    """
    <style>
    body {
        background-color: #2c3e50;
    }
    .stApp {
        background-color: #34495e;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Create input fields dynamically
user_inputs = {}
for feature in feature_names:
    user_inputs[feature] = st.number_input(f"{feature}", value=0.0)

# Convert user inputs to list
input_data = list(user_inputs.values())

if st.button("Predict"):
    result = predict_stroke(input_data)
    st.write(f"Prediction: {result}")
