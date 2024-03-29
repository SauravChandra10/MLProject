import sys
from src.MLProject.logger import logging
from src.MLProject.exception import CustomException

import numpy as np, pandas as pd


class DataTransformation:
    """
    This class handles data transformation steps for training and testing data.
    """
    def initiate_data_transormation(self,train_path,test_path):
        """
        Performs data transformation on training and testing CSV files.

        Args:
            train_path (str): The path to the training CSV file.
            test_path (str): The path to the testing CSV file.

        Returns:
            tuple: A tuple containing the transformed training and testing data as NumPy arrays.

        Raises:
            CustomException: If an error occurs during data transformation.
        """
        try:
            # Read training and testing data as pandas DataFrames
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            # logging.info("Reading the train and test file")

            # Create datetime, year, and day columns from the 'Date' column
            train_df['date']=pd.to_datetime(train_df['Date'], format = '%Y-%m-%d')
            train_df['Year'] = train_df['date'].dt.year
            train_df['Day'] = train_df['date'].dt.day

            test_df['date']=pd.to_datetime(test_df['Date'], format = '%Y-%m-%d')
            test_df['Year'] = test_df['date'].dt.year
            test_df['Day'] = test_df['date'].dt.day

            # Separate target feature from training and testing DataFrames
            target_feature_train_df=train_df['INR']
            train_df=train_df.drop(columns=['INR','Date','date'],axis=1)

            target_feature_test_df=test_df['INR']
            test_df=test_df.drop(columns=['INR','Date','date'],axis=1)

            # logging.info("Applying Preprocessing on training and test dataframe")

            # Combine features and target feature back into NumPy arrays
            train_arr = np.c_[train_df,np.array(target_feature_train_df)]
            test_arr = np.c_[test_df,np.array(target_feature_test_df)]

            return (
                train_arr,
                test_arr,
            )
        
        except Exception as e:
            raise CustomException(e,sys)

