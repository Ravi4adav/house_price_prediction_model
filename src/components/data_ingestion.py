# This file contains the code related to reading data

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import load_object

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('./notebooks/Data/Chennai houseing sale.csv')
            logging.info("Read the data as dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df,test_size=0.15,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Ingestion of data is completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e,sys)


# if __name__=='__main__':
#     dt_ingestion=DataIngestion()
#     train_data, test_data=dt_ingestion.initiate_data_ingestion()

#     train_data=pd.read_csv(train_data)
#     test_data=pd.read_csv(test_data)

#     dt_trans=DataTransformation()
#     train_data,test_data=dt_trans.get_transformed_data(test_data)
