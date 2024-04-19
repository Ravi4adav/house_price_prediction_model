import pandas as pd
import numpy as numpy
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer


pd.set_option('display.max_columns',None)


class DataTransformation:
    
    def __init__(self,train_data_path):
        self.data_transformation_config=DataTransformationConfig()
        self.train_data=pd.read_csv(train_data_path)
        self.obj_features=['AREA','SALE_COND', 'PARK_FACIL', 'BUILDTYPE', 'UTILITY_AVAIL', 'STREET', 'MZZONE']
        self.categorical_num_features=['N_BEDROOM', 'N_BATHROOM', 'N_ROOM']
        self.continuous_num_features=['INT_SQFT', 'DIST_MAINROAD', 'QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL', 'REG_FEE', 'COMMIS', 'SALES_PRICE']

#   Step-1:  
    def replace_multiple_val_to_single(self,data):
        try:
            self.data=data
            logging.info('Replacing multiple repeating values to a single class value')
            
            self.data.drop(['PRT_ID'],axis=1,inplace=True)
            self.val_dict={'AREA':{'Karapakkam':['Karapakam'], 'Anna Nagar':['Ana Nagar', 'Ann Nagar'], 'Adyar':['Adyr'], 'Velachery':['Velchery'], 
                        'Chrompet':['Chrompt', 'Chrmpet', 'Chormpet'], 'KK Nagar':['KKNagar'],'T Nagar': ['TNagar']},
                        'SALE_COND':{'AbNormal':['Ab Normal'], 'Partial':['Partiall', 'PartiaLl'], 'AdjLand':['Adj Land']},
                        'PARK_FACIL': {'No':['Noo']},
                        'BUILDTYPE':{'Commercial':['Comercial'], 'Others': ['Other']},
                        'UTILITY_AVAIL': {'AllPub':['All Pub'], 'NoSewa': ['NoSewr ','NoSwer','NoSeWa'], },
                        'STREET': {'Paved':['Pavd'], 'No Access':['NoAccess']}
                        }
            

            for feature in self.val_dict:
                for classes in self.val_dict[feature]:
                    for val in self.val_dict[feature][classes]:
                        self.data[feature]=self.data[feature].replace(val,classes)
            
            logging.info('Multiple repeating values to a single class value is replaced')
            return self.data

        except Exception as e:
            raise CustomException(e,sys)


#   Step-2
    def total_price_handling(self,data):
        try:
            self.tph_test_data=data
            logging.info("Initiating total price calculation")
            self.tph_test_data=self.replace_multiple_val_to_single(self.tph_test_data)
            self.tph_test_data['TOTAL_PRICE']=self.tph_test_data['COMMIS']+self.tph_test_data['REG_FEE']+self.tph_test_data['SALES_PRICE']
            self.tph_test_data.drop(['SALES_PRICE','COMMIS','REG_FEE'],axis=1,inplace=True)
            logging.info("Calculation of total price is done")

            return self.tph_test_data
        except Exception as e:
            raise CustomException(e,sys)


#   Step-3
    def encode_value(self,data):
        try:

            self.enc_test_data=data
            self.enc_test_data=self.total_price_handling(self.enc_test_data)
            self.train_data=self.total_price_handling(self.train_data)

            label_encoder=LabelEncoder()

            for feature in self.obj_features:
                label_encoder.fit(self.train_data[feature])
                self.enc_test_data[feature+'_enc']=label_encoder.transform(self.enc_test_data[feature])
                self.train_data[feature+'_enc']=label_encoder.transform(self.train_data[feature])
                self.enc_test_data.drop(feature,axis=1,inplace=True)
                self.train_data.drop(feature,axis=1,inplace=True)

            logging.info("Encoding of object datatype features is complete")
            return self.enc_test_data, self.train_data

        except Exception as e:
            raise CustomException(e,sys)

#   Step-4
    def temporal_feature_handling(self,data):
        try:
            self.temp_test_data=data
            logging.info("Initiating conversion of temporal features to numeric features")
            self.temp_test_data,self.train_data=self.encode_value(self.temp_test_data)

            self.train_data['DATE_BUILD']=pd.to_datetime(self.train_data['DATE_BUILD'],format='%d-%M-%Y')
            self.train_data['DATE_SALE']=pd.to_datetime(self.train_data['DATE_SALE'],format='%d-%M-%Y')
            self.train_data['HOUSE_AGE']=self.train_data['DATE_SALE'].dt.year-self.train_data['DATE_BUILD'].dt.year
            self.train_data.drop(['DATE_BUILD', 'DATE_SALE'],axis=1,inplace=True)


            self.temp_test_data['DATE_BUILD']=pd.to_datetime(self.temp_test_data['DATE_BUILD'],format='%d-%M-%Y')
            self.temp_test_data['DATE_SALE']=pd.to_datetime(self.temp_test_data['DATE_SALE'],format='%d-%M-%Y')
            self.temp_test_data['HOUSE_AGE']=self.temp_test_data['DATE_SALE'].dt.year-self.temp_test_data['DATE_BUILD'].dt.year
            self.temp_test_data.drop(['DATE_BUILD', 'DATE_SALE'],axis=1,inplace=True)

            logging.info("Temporal Features converted to Numeric feature")
            return self.temp_test_data,self.train_data
        except Exception as e:
            raise CustomException(e,sys)

#   Step-5
    def impute_missing_val(self,data):
        try:
            self.imputer_test_data=data
            logging.info("Initiating imputation of missing values")
            self.imputer_test_data, self.train_data=self.temporal_feature_handling(self.imputer_test_data)

            imputer=KNNImputer()
            self.imputer_test_data=pd.DataFrame(imputer.fit_transform(self.imputer_test_data),columns=self.imputer_test_data.columns)
            self.train_data=pd.DataFrame(imputer.fit_transform(self.train_data),columns=self.train_data.columns)

            logging.info("Imputation of missing values is complete")
            return self.imputer_test_data, self.train_data

        except Exception as e:
            raise CustomException(e,sys)



#   Step-6
    def scaling(self,data):
        try:
            self.scale_test_data=data
            logging.info("Initiating feature scaling")
            self.scale_test_data, self.train_data=self.impute_missing_val(self.scale_test_data)

            scale=StandardScaler()
            scale.fit(self.train_data)
            self.train_data=pd.DataFrame(scale.transform(self.train_data),columns=self.train_data.columns)
            self.scale_test_data=pd.DataFrame(scale.transform(self.scale_test_data),columns=self.scale_test_data.columns)
            logging.info("Feature Scaling is done")
            return self.train_data, self.scale_test_data
        except Exception as e:
            raise CustomException(e,sys)
        

    def get_transformed_data(self,data):
        try:
            self.data=data
            logging.info("Initiating Data Transformation")
            self.train_data, self.test_data=self.scaling(self.data)
            logging.info("Data transformation is done")
            return self.train_data, self.test_data

        except Exception as e:
            raise CustomException(e,sys)

