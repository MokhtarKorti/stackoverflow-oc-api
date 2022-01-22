from flask import Flask, request, jsonify
from utils import *


app = Flask(__name__)

@app.route('/')

def index():
    return ""
    

@app.route("/predict", methods=["GET"])
def predict():
    titre = request.form.get('title')
    corps = request.form.get('body')
    question = titre + ' ' + corps
    pred_tags = predict_tags(question)

    return  jsonify({"tags": pred_tags})

if __name__ == "__main__":
    app.run(debug=True)
