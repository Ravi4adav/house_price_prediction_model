import sys
import pandas as pd
import numpy as np
from src.utils import load_object
from src.exception import CustomException
from src.components.data_transformation import PredictDataTransformation



class PredictPipeline:
    def __init__(self):
        pass


    def predict(self, features):
        try:
            self.features=features
            print(self.features)
            model=load_object('./artifacts/model.pkl')
            preprocessor=PredictDataTransformation()
            self.features=preprocessor.preprocess(self.features)
            predict=model.predict(self.features)
            return predict
        except Exception as e:
            raise CustomException(e,sys)
        




class CustomData:
    def __init__(self, area: int, no_of_bedrooms: int, no_of_bathrooms: int,total_rooms: int, 
    region: str,building_type:str, property_zone:str,house_age: int):
        self.area=area
        self.no_of_bedrooms=no_of_bedrooms
        self.no_of_bathrooms=no_of_bathrooms
        self.total_rooms=total_rooms
        self.region=region
        self.building_type=building_type
        self.property_zone=property_zone
        self.house_age=house_age


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={"INT_SQFT": [self.area],"N_BEDROOM":[self.no_of_bedrooms],"N_BATHROOM":[self.no_of_bathrooms],
                                    'N_ROOM':[self.total_rooms],"AREA":self.region,"BUILDTYPE":self.building_type,
                                    "MZZONE":self.property_zone,"HOUSE_AGE":self.house_age}

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)