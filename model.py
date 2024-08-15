import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import json
data = pd.read_csv('logistic regression dataset-Social_Network_Ads.csv')

data['Age'] = data['Age'].apply(lambda x: int(x))
data['EstimatedSalary'] = data['EstimatedSalary'].apply(lambda x: int(x))
data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

X = data[['Age', 'Gender', 'EstimatedSalary']]
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
f2 = fbeta_score(y_test, y_test_pred, beta=2)

cm_df = pd.DataFrame(cm, index=['Not Purchased', 'Purchased'], columns=['Not Purchased', 'Purchased'])
cm_df.to_csv('confusion_matrix.csv')

metrics = {
    'Training Accuracy': train_accuracy,
    'Testing Accuracy': test_accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'F2 Score': f2
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

joblib.dump(model, "Logistic_Regression.pkl")
