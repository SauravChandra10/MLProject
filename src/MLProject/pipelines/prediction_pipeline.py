import sys, os
import io
from io import StringIO
import pandas as pd, numpy as np
from src.MLProject.exception import CustomException
from src.MLProject.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            model=load_object(file_path=model_path)
            preds=model.predict(features)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,f):
        stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        stream.seek(0)
        result = stream.read()
        df = pd.read_csv(StringIO(result))

        df['date'] = pd.to_datetime(df['Date'], format = '%d-%m-%Y')
        df['Year'] = df['date'].dt.year
        df['Day'] = df['date'].dt.day
        df=df.drop(columns=['Date','date'],axis=1)

        arr = np.array(df)

        self.arr=arr

    def get_data_as_data_frame(self):
        try:
            return self.arr

        except Exception as e:
            raise CustomException(e, sys)
        
class CustomDataJSON:
    def __init__(self,df):

        arr = np.array(df)

        self.arr=arr

    def get_data_as_data_frame(self):
        try:
            return self.arr

        except Exception as e:
            raise CustomException(e, sys)