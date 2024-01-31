import sys
from src.MLProject.logger import logging
from src.MLProject.exception import CustomException

import numpy as np, pandas as pd

from dataclasses import dataclass

@dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join('artifcats','preprocessor.pkl')

class DataTransformation:
    # def __init__(self):
    #     self.data_transformation_config=DataTransformationConfig()

    # def get_data_transformer_object(self):
    #     # this function is responsible for data transformation
    #     try:
    #         pass
    #     except Exception as e:
    #         raise CustomException (e,sys)
        
    def initiate_data_transormation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            train_df['date']=pd.to_datetime(train_df['Date'], format = '%Y-%m-%d')
            train_df['Year'] = train_df['date'].dt.year
            train_df['Day'] = train_df['date'].dt.day

            test_df['date']=pd.to_datetime(test_df['Date'], format = '%Y-%m-%d')
            test_df['Year'] = test_df['date'].dt.year
            test_df['Day'] = test_df['date'].dt.day

            target_feature_train_df=train_df['INR']
            train_df=train_df.drop(columns=['INR','Date','date'],axis=1)

            target_feature_test_df=test_df['INR']
            test_df=test_df.drop(columns=['INR','Date','date'],axis=1)

            logging.info("Applying Preprocessing on training and test dataframe")


            train_arr = np.c_[train_df,np.array(target_feature_train_df)]
            test_arr = np.c_[test_df,np.array(target_feature_test_df)]

            return (
                train_arr,
                test_arr,
            )
        
        except Exception as e:
            raise CustomException(e,sys)

