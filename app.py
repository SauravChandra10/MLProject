from flask import Flask, request, render_template, jsonify
import sys
from src.MLProject.exception import CustomException
from src.MLProject.pipelines.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.errorhandler(CustomException)
def handle_my_error(error):
    res={
        "status":False,
        "message":"Error!",
        "data":error
    }
    response = jsonify(res)
    return response

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        f = request.files['data_file']
        data = CustomData(f).arr
        prediction = PredictPipeline().predict(data)
        res={
            "status":True,
            "message":"Prediction done successfully!",
            "data":prediction[0]
        }
        return jsonify(res)
    except Exception as e:
        error = CustomException(e,sys).error_message
        return handle_my_error(error)

if __name__ == "__main__":
    app.run(port=5000)