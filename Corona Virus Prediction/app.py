import numpy as np
from flask import Flask, request, jsonify, render_template
from datetime import date
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb')) #confirm
model1 = pickle.load(open('model1.pkl', 'rb')) #death
model2 = pickle.load(open('model2.pkl', 'rb')) #recover

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    f_date=date(2020,1,22)
    features = [str(x) for x in request.form.values()]
    features = features[0]
    if re.match(r"[\d]{1,2}/[\d]{1,2}/[\d]{4}", features):
        features = features.split("/")
        
        if(!(int(features[1])<=4)):
        return render_template('index.html',prediction_text="Month should be upto April")  

        if(!(int(features[2])==2020)):
        return render_template('index.html',prediction_text="Year should be 2020")
    
        continue
        
    else:
        return render_template('index.html',prediction_text="Date should be in format dd/mm/yyyy")
    

    l_date=date(int(features[2]),int(features[1]),int(features[0]))
    delta = l_date - f_date
    prediction = model.predict([[delta.days]])
    prediction1 = model1.predict([[delta.days]])
    prediction2 = model2.predict([[delta.days]])

    return render_template('index.html', prediction_text="Total number of Confirmed Cases: {}".format(int(prediction)),prediction_text1="Total number of Death: {}".format(int(prediction1)),prediction_text2="Total number of Recover Patients: {}".format(int(prediction2)))
@app.route('/visual')
def visual():
    return render_template('visualize.html')
"""@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)"""

if __name__ == "__main__":
    app.run(debug=True)
