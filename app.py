import sys
from src.MLProject.logger import logging 
from src.MLProject.exception import CustomException 
from src.MLProject.components.data_ingestion import DataIngestion
from src.MLProject.components.data_transformation import DataTransformation
from src.MLProject.components.model_trainer import ModelTrainerConfig, ModelTrainer

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        train_path,test_path=data_ingestion.initiate_data_ingestion()
        
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transormation(train_path,test_path)

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)