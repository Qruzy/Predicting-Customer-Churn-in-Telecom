import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

df = pd.read_csv(r"C:\Users\User\Desktop\datascience\capstone\application\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.drop(['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'], axis='columns', inplace=True)

df['Churn'] = df.Churn.replace(('No', 'Yes'),(0, 1))
df['OnlineSecurity'] = df.OnlineSecurity.replace(('No internet service', 'Yes', 'No'),(2, 0, 1))
df['OnlineBackup'] = df.OnlineBackup.replace(('No internet service', 'Yes', 'No'),(2, 0, 1))
df['DeviceProtection'] = df.DeviceProtection.replace(('No internet service', 'Yes', 'No'),(2, 0, 1))
df['TechSupport'] = df.TechSupport.replace(('No internet service', 'Yes', 'No'),(2, 0, 1))

features = df.drop('Churn', axis = 1)
label = df['Churn']

scaler = MinMaxScaler()

features_scaled = scaler.fit_transform(features)

x_train, x_test, y_train, y_test = train_test_split(features_scaled, label, test_size=0.1, shuffle=True, random_state=2)

svc = SVC(probability = True)
svc.fit(x_train, y_train)

score = svc.score(x_test, y_test)
print(score)

joblib.dump(svc, r"C:\Users\User\Desktop\datascience\capstone\application\model\cust_churn.pkl")