import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from submodule.load_data import load_data
from submodule.train_model import train_model
from submodule.plot_graph import plot_graph

def main():
    st.title("Streamlit Dashboard")
    data = load_data()    
    page = st.sidebar.selectbox("Select a page:",["Homepage", "Exploration", "Modelling"])
    
    if page == "Homepage":
        st.title("Homepage")
        st.text("Customer Churn in Telecom Dataset")
        st.dataframe(data)
    elif page == "Exploration":
        fig1, fig2 = plot_graph(data)
        st.title("Exploratory Data Analysis")
        st.text("Correlation Coefficient")
        st.pyplot(fig1)
        st.text("Pairplot with Different Services")
        st.pyplot(fig2)
    else:
        st.title("Modelling")
        #Declare models
        log_reg = LogisticRegression()
        dtc = DecisionTreeClassifier()
        svc = SVC()
        rf_clf = RandomForestClassifier()
        knn = KNeighborsClassifier(n_neighbors = 2)
        xgb = XGBClassifier()
        
        models = [log_reg, dtc, svc, rf_clf, knn, xgb]
        for model in models:
            trained_model, score = train_model(model, data)
            st.write(f"{model} Model Accuracy is {score}")

if __name__=='__main__':
    main()