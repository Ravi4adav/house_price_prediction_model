from sklearn.feature_selection import SelectKBest, mutual_info_regression
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
import pandas as pd



class SelectFeature:
    def __init__(self, feature_num):
        self.feature_num=feature_num

    def feature_selection(self,x_train,y_train,x_test):
        self.x_train=x_train
        self.x_test=x_test
        self.k_features=SelectKBest(mutual_info_regression,k=self.feature_num)
        self.k_features.fit(x_train,y_train)
        self.k_features.transform(x_train)
        self.x_train=self.x_train[self.k_features.get_feature_names_out()]
        self.x_test=self.x_test[self.k_features.get_feature_names_out()]

        return self.x_train, self.x_test


# if __name__=="__main__":
#     dt_ingestion=DataIngestion()
#     train_data, test_data=dt_ingestion.initiate_data_ingestion()

#     test_data=pd.read_csv(test_data)

#     dt_trans=DataTransformation('./artifacts/train.csv')
#     train_data,test_data,y_train,y_test=dt_trans.get_transformed_data(test_data)

#     # print(train_data)

#     featureselection=SelectFeature(feature_num=8)
#     scale_train=featureselection.feature_selection(train_data,y_train)
#     print(scale_train)
