from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn

app = Flask(__name__)
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


def get_series(string):
    if '1' in string:
        return 1
    elif '2' in string:
        return 2
    elif '3' in string:
        return 3
    elif '4' in string:
        return 4
    elif '5' in string:
        return 5
    elif '6' in string:
        return 6
    elif '7' in string:
        return 7
    elif '8' in string:
        return 8


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Kms_Driven=int(request.form['Kms_Driven'])
        Tax = int(request.form['Tax'])
        Mpg = float(request.form['mpg'])
        Engine = float(request.form['engine'])
        Fuel_Type_Petrol=request.form['Fuel_Type_Petrol']
        if Fuel_Type_Petrol=='Petrol':
            Electric=0
            Hybrid=0
            Other=0
            Petrol=1
        elif Fuel_Type_Petrol=='Diesel':
            Electric=0
            Hybrid=0
            Other=0
            Petrol=0
        elif Fuel_Type_Petrol=='Hybrid':
            Electric=0
            Hybrid=1
            Other=0
            Petrol=0
        elif Fuel_Type_Petrol=='Electric':
            Electric=1
            Hybrid=0
            Other=0
            Petrol=0
        else:
            Electric=0
            Hybrid=0
            Other=1
            Petrol=0	
        Transmission_Mannual=request.form['Transmission_Mannual']
        if Transmission_Mannual=='Mannual':
            Transmission_Mannual=1
            Transmission_Semi=0
        elif Transmission_Mannual=='Automatic':
            Transmission_Mannual=0
            Transmission_Semi=0
        else:
            Transmission_Mannual=0
            Transmission_Semi=1
        Car_Model = request.form['Model']
        Series = get_series(Car_Model)
        if 'Series' in Car_Model:
            S = 1
            X = 0
            Z = 0
            i = 0
        elif 'X' in Car_Model:
            S = 0
            X = 1
            Z = 0
            i = 0
        elif 'i' in Car_Model:
            S = 0
            X = 0
            Z = 0
            i = 1
        elif 'Z' in Car_Model:
            S = 0
            X = 0
            Z = 1
            i = 0
        elif 'M' in Car_Model:
            S = 0
            X = 0
            Z = 0
            i = 0
        prediction=model.predict([[Year,Kms_Driven,Tax,Mpg,Engine,Transmission_Mannual,Transmission_Semi,Electric,Hybrid,Other,Petrol,Series,S,X,Z,i]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You Can Sell This Car at {}".format(output))
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)

