import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv("/Users/ajay/Desktop/PGA 36/Machine Learning/My projects/alzheimers project/Alzheimer_data.csv")
data.head(40)

X=data.drop('Group',axis=1)
y=data.Group


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model=RandomForestClassifier()
model.fit(X_train,y_train)



with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
    
    
with open('model.pkl', 'rb')as file:
    model=pickle.load(file)    


def predict(data):
    prediction=model.predict(data)
    return prediction    

import streamlit as st

st.title("Alzheimers  Disease Prediction ")

