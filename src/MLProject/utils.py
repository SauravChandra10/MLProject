import os
import sys 
from src.MLProject.exception import CustomException
from src.MLProject.logger import logging
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    """
    Saves a given Python object to a file using pickle.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj: The object to be saved.

    Raises:
        CustomException: If an error occurs during saving.
    """
    try:
        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in binary write mode and pickle the object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    """
    Evaluates multiple machine learning models using GridSearchCV and returns a report of their scores.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training target values.
        X_test (array-like): Testing features.
        y_test (array-like): Testing target values.
        models (dict): A dictionary containing model objects to evaluate, keyed by model names.
        param (dict): A dictionary of hyperparameter grids for each model, keyed by model names.

    Returns:
        dict: A dictionary containing model names as keys and their evaluation scores (RMSE) as values.

    Raises:
        CustomException: If an error occurs during evaluation.
    """

    try:
        report={}

        # Iterate through each model in the models dictionary
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            # Perform hyperparameter tuning using GridSearchCV
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            # Set the model's best parameters and refit on the entire training set
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # Make predictions on training and testing sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate RMSE on the testing set
            test_model_score = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Add the model name and score to the report dictionary
            report[list(models.keys())[i]] = test_model_score

        return report
        
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    """
    Loads a Python object from a file using pickle.

    Args:
        file_path (str): The path to the file containing the pickled object.

    Returns:
        The loaded Python object.

    Raises:
        CustomException: If an error occurs during loading.
    """
    try:
        # Open the file in binary read mode and unpickle the object
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)