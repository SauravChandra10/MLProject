from flask import Flask, request, render_template, jsonify
import sys
from src.MLProject.exception import CustomException
from src.MLProject.pipelines.prediction_pipeline import PredictPipeline, CustomData, CustomDataJSON
from src.MLProject.components.data_ingestion import DataIngestion
from src.MLProject.components.data_transformation import DataTransformation
from src.MLProject.components.model_trainer import ModelTrainer
import pandas as pd
import io
from io import StringIO
from werkzeug.exceptions import RequestEntityTooLarge
from middleware import auth

application = Flask(__name__)

app = application

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 # 5 MB


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
@auth
def index():
    return render_template("index.html")

@app.route('/trainJSON',methods=["POST"])
def trainJSON():
    try:
        data = request.json

        if not data or not isinstance(data, list):
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        df = pd.DataFrame.from_dict(data, orient='columns')

        obj=DataIngestion()
        train_data_path,test_data_path=obj.initiate_data_ingestion(df)

        data_transformation = DataTransformation()
        train_arr,test_arr = data_transformation.initiate_data_transormation(train_data_path,test_data_path)

        modeltrainer = ModelTrainer()
        print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

        res={
            "status":True,
            "message":"Training done successfully!",
            "data":"model created at artifacts/model.pkl"
        }
        return jsonify(res)

    except Exception as e:
        error = CustomException(e,sys).error_message
        return handle_my_error(error)

@app.route('/predictJSON', methods=['POST'])
def predictJSON():
    try:
        data = request.json

        if not data or not isinstance(data, list):
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        df = pd.DataFrame.from_dict(data, orient='columns')

        data = CustomDataJSON(df).arr

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

@app.route('/train',methods=["POST"])
@auth
def train():
    try:
        f=request.files['train_file']

        if not f:
            res={
                "status":False,
                "message":"Error!",
                "data":"No file uploaded"
            }
            return res, 400  
        
        filename = f.filename
        if not filename.endswith('.csv'):
            res={
                "status":False,
                "message":"Error!",
                "data":"Only CSV files allowed"
            }
            return res, 400
        
        stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        stream.seek(0)
        result = stream.read()
        df = pd.read_csv(StringIO(result))
        obj=DataIngestion()
        train_data_path,test_data_path=obj.initiate_data_ingestion(df)

        data_transformation = DataTransformation()
        train_arr,test_arr = data_transformation.initiate_data_transormation(train_data_path,test_data_path)

        modeltrainer = ModelTrainer()
        print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

        res={
            "status":True,
            "message":"Training done successfully!",
            "data":"model created at artifacts/model.pkl"
        }
        return jsonify(res)
    
    except RequestEntityTooLarge as e:
        res = {
            "status": False,
            "message": "Error!",
            "data": 'File size exceeds 5 MB'
        }
        return res, 413
    except Exception as e:
        error = CustomException(e,sys).error_message
        return handle_my_error(error)
    
@app.route('/predict', methods=["POST"])
def predict():
    try:
        f = request.files['test_file']

        if not f:
            res={
                "status":False,
                "message":"Error!",
                "data":"No file uploaded"
            }
            return res, 400  

        filename = f.filename
        if not filename.endswith('.csv'):
            res={
                "status":False,
                "message":"Error!",
                "data":"Only CSV files allowed"
            }
            return res, 400

        data = CustomData(f).arr
        prediction = PredictPipeline().predict(data)
        res={
            "status":True,
            "message":"Prediction done successfully!",
            "data":prediction[0]
        }
        return jsonify(res)
    except RequestEntityTooLarge as e:
        res = {
            "status": False,
            "message": "Error!",
            "data": 'File size exceeds 1 MB'
        }
        return res, 413
    except Exception as e:
        error = CustomException(e,sys).error_message
        return handle_my_error(error)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)