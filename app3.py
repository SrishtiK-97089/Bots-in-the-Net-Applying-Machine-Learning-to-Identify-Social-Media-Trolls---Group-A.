# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 07:29:49 2021

@author: HP PC
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
# Keras

from keras.models import load_model


clf=load_model("LSTM_TeamA_model.h5")


cv=pickle.load(open('vectorizer.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
  if request.method == 'POST':
   message = request.form['message']
   data = [message]
   vect = cv.transform(data).toarray()
   my_prediction = clf.predict(vect)
  return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
  app.run(debug=True)