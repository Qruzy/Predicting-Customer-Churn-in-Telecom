import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

def plot_graph(data):
    fig1, ax = plt.subplots(figsize=(10,10))
    le = LabelEncoder()
    data['Churn'] = le.fit_transform(data['Churn'])
    matrix = data.corr()
    sns.heatmap(matrix, annot=True, ax=ax)
    
    fig2 = sns.pairplot(data, vars = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport'], hue = 'Churn')
    
    return fig1, fig2