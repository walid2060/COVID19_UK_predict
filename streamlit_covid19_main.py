import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from keras.models import load_model

import numpy as np
import pandas as pd
import  pickle
loaded_model = pickle.load(open(r'C:\Users\HP\Desktop\finalized_model_Polynomial_covid19.sav','rb'))

def predict_malade():
    st.title(' prediction of New cases infected WITH Covid19 ')
    
    # Input form
    st.subheader('Global Information')
    patient = {}
    patient['confirmedRate'] = st.number_input('donner le nombres des cas confirmes')
    patient['confirmed'] = st.number_input('donner le nombre des cas confirmes')
    patient['deathNew'] = st.number_input('donner le nombre des cas nouveau morts')
# Preprocess input data
    patient_df = pd.DataFrame(patient, index=[0])
    
    # Make prediction
    if st.button('Predict New cases infected ON Covid19'):
            poly=PolynomialFeatures(degree=5)
            #poly.fit(x_train, y_train)
            #x_train_fit = poly.fit_transform(x_train) #transforming our input data
            #model.fit(x_train_fit, y_train)
            x_test_ = poly.fit_transform(patient_df)

            #loaded_model = pickle.load(open(r'C:\Users\HP\Desktop\finalized_model_Polynomial_covid19.sav','rb'))
            prediction = loaded_model.predict(x_test_)
    
            
            st.write('number of new cases infected with Covid19 is :',prediction[0])
if __name__ == '__main__':
    predict_malade()
                   
