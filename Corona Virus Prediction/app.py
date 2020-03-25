import numpy as np
from flask import Flask, request, jsonify, render_template
from datetime import date
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))

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
    features = features.split("/")
    

    l_date=date(int(features[2]),int(features[1]),int(features[0]))
    delta = l_date - f_date
    prediction = model.predict([[delta.days]])
    prediction1 = model1.predict([[delta.days]])
    prediction2 = model2.predict([[delta.days]])

    return render_template('index.html', prediction_text=int(prediction1))
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
