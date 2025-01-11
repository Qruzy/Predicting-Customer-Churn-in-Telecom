from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

def train_model(model, data):
    features = data.drop('Churn', axis = 1)
    label = data['Churn']
    
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    x_train, x_test, y_train, y_test = train_test_split(features_scaled, label, test_size=0.1, shuffle=True, random_state=2)
    
    model.fit(x_train, y_train)
    
    score = model.score(x_test, y_test)
    
    return model, score