import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model=load_model('model.h5')

with open('geo_encoder.pkl','rb') as file:
    onehot=pickle.load(file)

with open('label_encode_gender.pkl','rb') as file:
    lable_encoder=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


st.title('Customer Churn prediction')

geoghraphy=st.selectbox('Geography',onehot.categories_[0])
gender=st.selectbox('Gender',lable_encoder.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
Credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimate Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr=st.selectbox('Has Credit Card',[0,1])
is_active=st.selectbox('IsActive',[0,1])


input_data = pd.DataFrame({
    'CreditScore': [Credit_score],
    'Gender': [lable_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr],
    'IsActiveMember': [is_active],
    'EstimatedSalary': [estimated_salary]
})

df=pd.DataFrame(onehot.transform([[geoghraphy]]).toarray(),columns=onehot.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data,df],axis=1)

scaleinput=scaler.transform(input_data)

pridict=model.predict(scaleinput)
prob=pridict[0][0]

st.write(prob)

if prob>0.5:
    st.write('the customer is likely to churn')
else:
    st.write('the customer is not likely to churn')