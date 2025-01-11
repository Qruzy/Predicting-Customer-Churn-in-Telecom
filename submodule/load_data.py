import pandas as pd

def load_data():
    df = pd.read_csv(r"C:\Users\User\Desktop\datascience\capstone\application\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df.drop(['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'], axis='columns', inplace=True)

    df['Churn'] = df.Churn.replace(('No', 'Yes'),(0, 1))
    df['OnlineSecurity'] = df.OnlineSecurity.replace(('No internet service', 'Yes', 'No'),(2, 0, 1))
    df['OnlineBackup'] = df.OnlineBackup.replace(('No internet service', 'Yes', 'No'),(2, 0, 1))
    df['DeviceProtection'] = df.DeviceProtection.replace(('No internet service', 'Yes', 'No'),(2, 0, 1))
    df['TechSupport'] = df.TechSupport.replace(('No internet service', 'Yes', 'No'),(2, 0, 1))
    
    return df