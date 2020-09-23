# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:40:58 2020

@author: rhitc
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("Dream Housing Finance company")
st.header("Home Loan Approval Prediction")
st.markdown("Enter the following parameters from the sidebar")

gender=st.sidebar.selectbox("Gender",['Male','Female'])

married=st.sidebar.radio("Marital Status",['Married','Unmarried'])

dependents=st.sidebar.select_slider("Number of dependents",['0','1','2','3+'])

education=st.sidebar.selectbox('Education level',['Graduate','Not Graduate'])

self_emp=st.sidebar.radio("Self Employed",['Yes','No'])

income=st.sidebar.number_input("Enter annual income(LPA)")

c_income=st.sidebar.number_input("Co-applicant's income(LPA)")

loan=st.sidebar.number_input("Loan Amount(in thousands)")

loan_term=st.sidebar.selectbox("Term of loan (in months)",[12.0,36.0,60.0,84.0,120.0,180.0,240.0,300.0,360.0,480.0])

p_area=st.sidebar.selectbox("Property Area",['Urban', 'Rural', 'Semiurban'])

cred_hist=st.sidebar.radio("Credit history",['Good','Bad'])



dic={
'Gender':gender,
'Marital Status':married,
'Dependents':dependents,
'Education':education,
'Self Employed':self_emp,
'ApplicantIncome':income,
'CoapplicantIncome':c_income,
'LoanAmount':loan,
'Loan_Amount_Term':loan_term,
'Property_Area':p_area,
'Credit History':cred_hist}

st.dataframe(data=pd.DataFrame(dic,index=[0]))


#loading model
def load_models():
    models={
        'le_dep':pickle.load(open("Models/le_Dependents.pkl","rb")),
        'le_edu':pickle.load(open("Models/le_Education.pkl","rb")),
        'le_loan_terms':pickle.load(open("Models/le_Loan_Amount_Term.pkl","rb")),
        'le_prop_ar':pickle.load(open("Models/le_Property_Area.pkl","rb")),
        
        'sc_income':pickle.load(open("Models/sc_ApplicantIncome.pkl","rb")),
        'sc_co_income':pickle.load(open("Models/sc_CoapplicantIncome.pkl","rb")),
        'sc_amt':pickle.load(open("Models/sc_LoanAmount.pkl","rb")),
        
        'clf_log':pickle.load(open("Models/clf_log.pkl","rb")),
        'clf_rf':pickle.load(open("Models/clf_rf.pkl","rb")),
        'clf_svm':pickle.load(open("Models/clf_svm.pkl","rb")),
        'clf_xgb':pickle.load(open("Models/clf_xgb.pkl","rb"))
}
    return models

models=load_models()

#feature engineering
if gender=='Male':
    Gender_Male=1
    Gender_Female=0
else:
    Gender_Male=0
    Gender_Female=1
    
if married=='Married':
    mar_yes=1
    mar_no=0
else:
    mar_yes=0,
    mar_no=1
    

if self_emp=='Yes':
    semp_y=1
    semp_n=0
else:
    semp_y=0
    semp_n=1
    

income=income*100
c_income=c_income*100

if cred_hist=='Good':
    c_good=1
    c_bad=0
else:
    c_good=0
    c_bad=1


#lable encoding
dependents=models['le_dep'].transform([dependents])
education=models['le_edu'].transform([education])
loan_term=models['le_loan_terms'].transform([loan_term])
p_area=models['le_prop_ar'].transform([p_area])

#Sclaing
income=models['sc_income'].transform([[income]])[0][0]
c_income=models['sc_co_income'].transform([[c_income]])[0][0]
loan=models['sc_amt'].transform([[loan]])[0][0]

dict_pred={'Dependents':dependents,
'Education':education,
'ApplicantIncome':income,
'CoapplicantIncome':c_income,
'LoanAmount':loan,
'Loan_Amount_Term':loan_term,
'Property_Area':p_area,
'Gender:Female':Gender_Female,
'Gender:Male':Gender_Male,
'Married:No':mar_no,
'Married:Yes':mar_yes,
'Self_Employed:No':semp_n,
'Self_Employed:Yes':semp_y,
'Credit_History:0.0':c_bad,
'Credit_History:1.0':c_good
}

df_pred=pd.DataFrame(dict_pred)

#prediction
sample=pd.DataFrame()
sample['logistic Regression']=models['clf_log'].predict_proba(df_pred)[:,1]
sample['SVM']=models['clf_svm'].predict_proba(df_pred)[:,1]
sample['XGBoost']=models['clf_xgb'].predict_proba(df_pred)[:,1]
sample['Random Forest']=models['clf_rf'].predict_proba(df_pred)[:,1]

rate=sample[['logistic Regression','SVM','XGBoost','Random Forest']].values.mean(axis=1)[0]

if st.button('Predict'):
    st.subheader("Chance of approval according to different algoriths")
    st.dataframe(sample*100)
    st.write(f'Chance of approval of Home Loan {round(rate*100,2)}%')
    if rate > 0.5:
        st.success("yeah!! Good Chance of approval")
    else:
        st.error(" Oops!! Bad chance of approval")
else:
    st.write('Check your eligibility')
