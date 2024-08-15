import streamlit as slt
import joblib
from sklearn.linear_model import LogisticRegression
import pandas as pd
import json 

slt.title('Purchasing Probability Predictor')
slt.write('Enter the values and algorithm will calculate whether you will purchase something from social network ads')

model = joblib.load("Logistic_Regression.pkl")

Age = slt.number_input("Enter your age: ")
Gender = slt.selectbox("Select your Gender (1 for Male, 0 for Female)", options=[0,1])
Salary = slt.number_input("Enter your estimated Salary:", min_value=0, step=1000, format="%d")

input_data = {
    'Age' : [Age], 'Gender' : [Gender], 'EstimatedSalary' : [int(Salary)]  
}

data = pd.DataFrame(input_data)

data.columns = data.columns.astype(str)

result = model.predict(data)

if slt.button('Calculate'):
    slt.write("Whether you will purchase something from social media ad:")
    if result == 1:
        slt.write("YES")
    else:
        slt.write("NO")

slt.title("Info Regarding our Algorithm")
cm_df = pd.read_csv('confusion_matrix.csv', index_col=0)
slt.write("Confusion Matrix:")
slt.write(cm_df)
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

slt.write(f"Training Accuracy: {metrics['Training Accuracy']:.4f}")
slt.write(f"Testing Accuracy: {metrics['Testing Accuracy']:.4f}")
slt.write(f"Precision: {metrics['Precision']:.4f}")
slt.write(f"Recall: {metrics['Recall']:.4f}")
slt.write(f"F1 Score: {metrics['F1 Score']:.4f}")
slt.write(f"F2 Score: {metrics['F2 Score']:.4f}")
