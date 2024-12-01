from flask import Flask, jsonify, render_template,request
from sklearn.preprocessing import StandardScaler
import pickle


app = Flask(__name__)

ridge_model = pickle.load(open('ridge.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
@app.route("/")
def hello_world():
    return render_template('index.html')



@app.route("/predictdata",methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
       
        scaled_data = scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(scaled_data)
        print(result)

        return render_template('home.html',result = result[0])

    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

