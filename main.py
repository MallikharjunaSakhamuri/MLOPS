from flask import Flask
from flask import request
from joblib import load
from flask import jsonify

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World from Flask"

@app.route("/predict",methods=['POST'])
def predict():
    data = request.get_json()
    values  = list(data.values())
    print("values",values)
    model = load("WineQuality.pkl")
    prediction = model.predict([values])
    return jsonify({'result':prediction[0]})

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=80)
