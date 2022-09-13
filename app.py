# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 17:22:11 2022

@author: john.l.a.zabanal
"""
from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

model = load_model('deployment_28042020')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

#def run():
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
    
image = Image.open('logo.png')
image_hospital = Image.open('hospital.jpg')
#print(image)

st.image(image,use_column_width=False)

add_selectbox = st.sidebar.selectbox(
"How would you like to predict?",
("Online", "Batch"))

st.sidebar.info('This app is created to predict patient hospital charges')
st.sidebar.success('https://www.pycaret.org')

st.sidebar.image(image_hospital)

st.title("Insurance Charges Prediction App")

if add_selectbox == 'Online':

    age = st.number_input('Age', min_value=1, max_value=100, value=25)
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
    children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
    if st.checkbox('Smoker'):
        smoker = 'yes'
    else:
        smoker = 'no'
    region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

    output=""

    input_dict = {'age' : age, 'sex' : sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region' : region}
    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        output = '$' + str(output)

    st.success('The output is {}'.format(output))

if add_selectbox == 'Batch':

    file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

    if file_upload is not None:
        data = pd.read_csv(file_upload)
        predictions = predict_model(estimator=model,data=data)
        st.write(predictions)
        
        #Download data
        csv = convert_df(predictions)    
        
        st.download_button(
         label="Download data as CSV",
         data=csv,
         file_name='lau_test_df.csv',
         mime='text/csv'
        )
#if __name__ == '__main__':
#    run()
