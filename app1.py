# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 10:55:26 2021

@author:SOUPTIK
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("classifier_RF.pkl", 'rb'))
cv=pickle.load(open("vectorizer.pkl",'rb'))




def predict_default(features):
    
    features = cv.fit(features)
    features = np.array(features).reshape(-1,1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    return prediction, probability


def main():

    st.title("Detecting Twitter Bots")
    st.markdown("""
    <div style = "background-color: green; padding: 10px;">
    body {
    color: pink;
    background-color: yellow;
    <center><h1><i>Twitter Bots Prediction<i></h1></center>
    }
    </div>
        """, unsafe_allow_html=True)

    NAME = st.text_input("Name")
    
    
    
    
    TWEETS = st.text_input("TWEETS","Please Enter your tweet here")

    if st.button("Predict"):
        features = [TWEETS]
        prediction, probability = predict_default(features)
        # print(prediction)
        # print(probability[:,1][0])
        if prediction[0] == 1:
            # counselling_html = """
            #     <div style = "background-color: #f8d7da; font-weight:bold;padding:10px;border-radius:7px;">
            #         <p style = 'color: #721c24;'>This account will be defaulted with a probability of {round(np.max(probability)*100, 2))}%.</p>
            #     </div>
            # """
            # st.markdown(counselling_html, unsafe_allow_html=True)

            st.success("This tweet is made by Human ")

        else:
            st.success("This tweet is made by bot")




if __name__ == '__main__':
    main()