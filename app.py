import streamlit as st
import pickle
import  sklearn
import pandas as pd
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

st.title('Business Sales prediction')
st.sidebar.header('Business Data')

images = ['TV.jpg', 'Radio.jpg' , 'Newspaper.jpg']
st.image(images)

def user_report():
    TV= st.sidebar.slider('TV', 1,300,1)
    Radio = st.sidebar.slider('Radio', 1, 100, 1)
    Newspaper = st.sidebar.slider('Newspaper', 1, 100, 1)

    user_report_data= {'TV' : TV, 'Radio' : Radio, 'Newspaper' : Newspaper }
    report_data= pd.DataFrame(user_report_data, index=[0])
    return report_data
user_data= user_report()
st.header('Business Data')
st.write(user_data)

Sales= model.predict(user_data)
st.subheader('Sales Amount')
st.subheader(Sales)



