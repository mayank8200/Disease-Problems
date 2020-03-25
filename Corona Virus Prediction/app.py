import numpy as np
from flask import Flask, request, jsonify, render_template
from datetime import date
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
f_date = date(2020, 1, 22)
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features1=[]
    int_features = request.form.values()
    int_features = int_features.split("/")
    for i in int_features:
       int_features1.append(int(i)) 

    l_date=date(int_features1[2],int_features1[1],int_features1[0])
    delta = l_date - f_date
    final_features = [np.array(delta)]
    prediction = model.predict(final_features)

    output = round(prediction[0]*100000, 2)

    return render_template('index.html', prediction_text='House Price Should Be Rs {}'.format(output),prediction_text1 = "Hello",prediction_text2 = "World")
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
