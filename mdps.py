# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 20:36:31 2025

@author: Shahi
"""

import pickle 
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# Load the saved models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# ✅ Load the scaler used during training for diabetes model
scaler_diabetes = pickle.load(open('scaler_diabetes.sav', 'rb'))
scaler_heart_disease = pickle.load(open('scaler_heart_disease.sav', 'rb'))
scaler_parkinsons = pickle.load(open('scaler_parkinsons.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        '🩺Multiple Disease Prediction System using ML',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# ==================== Diabetes Prediction Page ====================
if selected == 'Diabetes Prediction':
    
    # Page title
    st.title('🧪 Diabetes Prediction')

    # Input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
        
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
        
    with col1:    
        SkinThickness = st.text_input('Skin Thickness value')
        
    with col2:
        Insulin = st.text_input('Insulin Level')
       
    with col3:
        BMI = st.text_input('BMI value')
       
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        
    with col2:
        Age = st.text_input('Age of the Person')

    # Code for prediction
    diab_diagnosis = ''

    # Prediction button
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to float
            input_data = [[
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]]

            # ✅ Standardize input
            input_data_scaled = scaler_diabetes.transform(input_data)

            # ✅ Predict
            diab_prediction = diabetes_model.predict(input_data_scaled)

            # Output
            if diab_prediction[0] == 0:
                diab_diagnosis = '✅ The person is Not Diabetic'
            else:
                diab_diagnosis = '⚠️ The person is Diabetic'

        except ValueError:
            diab_diagnosis = '⚠️ Please enter valid numerical values in all fields.'

        st.success(diab_diagnosis)

# ==================== Heart Disease Prediction Page ====================
if selected == 'Heart Disease Prediction':
    
    # page title 
    st.title('❤️ Heart Disease Prediction')  
    
    # Input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.text_input('Age')
        
    with col2:
       Sex = st.text_input('Sex')
        
    with col3:
        Cp = st.text_input('Chest Pain types')
        
    with col1:    
      Trestbps = st.text_input('Resting Blood Pressure')
         
    with col2:
       Chol = st.text_input('Serum Cholestoral in mg/dl')
       
    with col3:
        FBS = st.text_input('Fasting Blood Sugar > 120 mg/dl')
       
    with col1:
        Restecg = st.text_input('Resting Electrocardiographic result')
        
    with col2:
       Thalach = st.text_input('Maximum Heart Rate achieved')
       
    with col3:    
       Exang = st.text_input('Exercise Induced Angina')
          
    with col1:
        Oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2 :
         Slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3 :
         CA = st.text_input('Major vesseles colored by flourosopy')
         
    with col1:
        Thal = st.text_input('thal : 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
    
        
    # Code for prediction
    diab_diagnosis = ''

    # Prediction button
    if st.button('Heart Disease Test Result'):
        try:
            # Convert inputs to float
            input_data = [[
                float(Age), float(Sex), float(Cp), float(Trestbps),
                float(Chol), float(FBS), float(Restecg),
                float(Thalach), float(Exang), float(Oldpeak), float(Slope), float(CA),
                 float(Thal)
            ]]
            
            
            # ✅ Standardize input
            input_data_scaled = scaler_heart_disease.transform(input_data)

            # ✅ Predict
            diab_prediction = heart_disease_model.predict(input_data_scaled)

            
            # Output
            if diab_prediction[0] == 0:
                diab_diagnosis = '✅ The person does NOT have Heart Disease'
            else:
                diab_diagnosis = '⚠️ The person HAS Heart Disease'

        except ValueError:
            diab_diagnosis = '⚠️ Please enter valid numerical values in all fields.'

        st.success(diab_diagnosis)
    


# ==================== Parkinson's Prediction Page ====================
if selected == 'Parkinsons Prediction':
    
    # page title 
    st.title('🧠 Parkinsons Prediction')  
    
    # input fields
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        Fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
       Fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        Flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:    
      Jitter_percent = st.text_input('MDVP:Jitter(%)')
         
    with col5:
       Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
       
    with col1:
        RAP = st.text_input('MDVP:RAP')
       
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
       DDP = st.text_input('Jitter:DDP')
       
    with col4:    
       Shimmer = st.text_input('MDVP:Shimmer')
          
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1 :
         APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2 :
         APQ5 = st.text_input('Shimmer:APQ5')
         
    with col3:
        APQ = st.text_input('MDVP:APQ') 
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
       NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:    
      RPDE = st.text_input('RPDE')
         
    with col3:
       DFA = st.text_input('DFA')
       
    with col4:
        spread1 = st.text_input('spread1')
       
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
       D2 = st.text_input('D2')
       
    with col2:    
       PPE = st.text_input('PPE')
       
       
       
       
    # Code for prediction
    diab_diagnosis = ''

    # Prediction button
    if st.button('Heart Disease Test Result'):
        try:
            # Convert inputs to float
            input_data = [[
                float(Fo), float(Fhi), float(Flo), float(Jitter_percent),
                float(Jitter_Abs), float(RAP), float(PPQ),
                float(DDP), float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5),
                 float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE), float(DFA), 
                 float(spread1), float(spread2), float(D2), float(PPE)
            ]]
            
            
            # ✅ Standardize input
            input_data_scaled = scaler_parkinsons.transform(input_data)

            # ✅ Predict
            diab_prediction = parkinsons_model.predict(input_data_scaled)

            
            # Output
            if diab_prediction[0] == 0:
                diab_diagnosis = '✅ The person does NOT have Parkinson’s Disease'
            else:
                diab_diagnosis = '⚠️ The person HAS Parkinson’s Disease'

        except ValueError:
            diab_diagnosis = '⚠️ Please enter valid numerical values in all fields.'

        st.success(diab_diagnosis)  