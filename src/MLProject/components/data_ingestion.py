import os
import sys 
import pandas as pd
from src.MLProject.exception import CustomException
from src.MLProject.logger import logging
from src.MLProject.components.data_transformation import DataTransformation
from src.MLProject.components.model_trainer import ModelTrainer
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join('artifacts','raw.csv')
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # code for reading data 
            # df=read_sql_data()

            df = pd.read_csv('FX-SAMPLE-TRAIING-22JAN2024VER1.csv')

            logging.info("Reading data")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr = data_transformation.initiate_data_transormation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))