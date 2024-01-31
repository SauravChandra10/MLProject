import os, sys
from src.MLProject.logger import logging
from src.MLProject.exception import CustomException

from sklearn.ensemble import RandomForestRegressor

import numpy as np, pandas as pd

from src.MLProject.utils import save_object, evaluate_models

from dataclasses import dataclass

from sklearn.metrics import mean_squared_error

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split train and test input data")

            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                "RandomForest": RandomForestRegressor()
            }

            params={
                "RandomForest":{
                    'n_estimators':[400],
                    'min_samples_split':[2],
                    'min_samples_leaf':[1],
                    'max_features':['sqrt'],
                    'max_depth':[None],
                    'bootstrap':[False]
                }
            }

            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models,params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            return np.sqrt(mean_squared_error(y_test,predicted))

        except Exception as e:
            raise CustomException(e,sys)