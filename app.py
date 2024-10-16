import pickle
from flask import Flask
from flask import render_template , jsonify , request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Note the directory is down so will do models/ b/c the current file in on out side(parent of it)
with open("models/scaler.pkl" , 'rb') as f:
    scaler_model = pickle.load(f)

with open("models/Ridge.pkl" , 'rb') as f:  
    ridge_model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def  index():
    return render_template("index.html")


@app.route("/predictdata" , methods =["GET", "POST"])  # use to predict the user input data
def  predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        new_scaled_data = scaler_model.transform([[Temperature , RH ,Ws , Rain , FFMC , DMC , ISI , Classes ,Region]])

        result = ridge_model.predict( new_scaled_data )  # reslut will e list

        return render_template("home.html" , result = result[0])  # send ist element of list. here list is result


    else:
        return render_template("home.html")




if __name__ == "__main__":
    app.run(host='0.0.0.0')