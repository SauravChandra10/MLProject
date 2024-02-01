from flask import Flask, request, render_template
import sys
from src.MLProject.exception import CustomException
from src.MLProject.pipelines.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        f = request.files['data_file']
    except Exception as e:
        raise CustomException(e,sys)

    data = CustomData(f).df

    prediction = PredictPipeline().predict(data)

    return render_template("index.html",prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)