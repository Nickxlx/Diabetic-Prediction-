from flask import Flask, request, app, render_template, Response
import pickle
import pandas as pd
import numpy as np

scaler = pickle.load(open("Model/scaler.pkl", "rb"))
model = pickle.load(open("Model/logisticmodel.pkl","rb"))

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods= ['GET', 'POST'])
def predict_datapoint():

    result = ""

    if request.method == 'POST':
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        data = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

        predict = model.predict(data)[0]

        if predict == 1 :
            result =   "Diabetic"
        else:
            result =  "Not Diabetic"
        
        return render_template("result.html", result = result)

    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")
