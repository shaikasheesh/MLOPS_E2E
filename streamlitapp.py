import streamlit as st
import joblib
import os
path = os.path.join('models/RandomForest')
model = joblib.load(path)

def prediction(Joining_Year,PaymentTier,Age,ExperienceInCurrentDomain,Education_Masters,Education_PHD,City_New_Delhi,
                City_Pune,Gender_Male,EverBenched_Yes):
    return model.predict([[Joining_Year,PaymentTier,Age,ExperienceInCurrentDomain,Education_Masters,Education_PHD,City_New_Delhi,
                City_Pune,Gender_Male,EverBenched_Yes]])

def pipeline():
    st.title('Employee Churn Application Setup ')
    Joining_Year= st.text_input('Joining Year')
    PaymentTier = st.text_input('PaymentTier')
    Age = st.text_input('Age')
    ExperienceInCurrentDomain = st.text_input('ExperienceInCurrentDomain')
    Education_Masters = st.text_input('Education_Masters')
    Education_PHD = st.text_input('Education_PHD')
    City_New_Delhi = st.text_input('City_New_Delhi')
    City_Pune = st.text_input('City_Pune')
    Gender_Male = st.text_input('Gender_Male')
    EverBenched_Yes = st.text_input('EverBenched_Yes')
    result = ''
    if st.button('predict'):
        result = prediction(Joining_Year,PaymentTier,Age,ExperienceInCurrentDomain,Education_Masters,Education_PHD,City_New_Delhi,
                City_Pune,Gender_Male,EverBenched_Yes)
    st.success('the output of the result is {}'.format(result))

if __name__ == '__main__':
    pipeline()
