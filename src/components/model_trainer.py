import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle as pkl
from src.utils import save_object
import warnings

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.feature_select import SelectFeature

warnings.filterwarnings('ignore')


# dt_ingestion=DataIngestion()
# train_data, test_data=dt_ingestion.initiate_data_ingestion()

# test_data=pd.read_csv(test_data)

# dt_trans=DataTransformation('./artifacts/train.csv')
# train_data,test_data,y_train,y_test=dt_trans.get_transformed_data(test_data)

# # print(train_data)

# featureselection=SelectFeature(feature_num=8)
# x_train, x_test=featureselection.feature_selection(train_data,y_train,test_data)

# print(x_train.head())



model={
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso': Lasso(),
    'Support Vector Machine': SVR(),
    'KNN Regressor': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boost Regressor': GradientBoostingRegressor(),
    'AdaBoost Regressor': AdaBoostRegressor(),
    'XGBoost Regressor': XGBRegressor()
}


params={
    'Linear Regression': {},
    'Ridge Regression':{'alpha':[6,7,8,9], 'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']},
    'Lasso': {'alpha':[8,9,10]},
    'Support Vector Machine': {'C':[9000000,20000000],'kernel':['rbf']},
    'KNN Regressor': {'n_neighbors':[13,14,15],'weights':['uniform','distance'], 'algorithm':['auto','ball_tree','kd_tree','brute']},
    'Decision Tree':{'max_depth':[8,9,10], 'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    'splitter':['best','random'], },
    'Random Forest Regressor':{'max_depth':[9,10,11], 'n_estimators':[300,400]},
    'Gradient Boost Regressor':{ 'max_depth':[3,4,5],'n_estimators':[150,200]},
    'AdaBoost Regressor':{'n_estimators':[300,400,500],'learning_rate':[0.3,0.4,0.5]},
    'XGBoost Regressor':{'learning_rate':[0.09,0.1,0.2], 'max_depth':[4,5,6],'n_estimators':[200,300]}
    
}

def model_train(models: dict, params: dict):
    best_score=0.70
    best_model=""
    for models in model:
        reg_model=GridSearchCV(model[models], param_grid=params[models],scoring='neg_mean_squared_error',cv=10)
        reg_model.fit(x_train,y_train)
        reg_model=reg_model.best_estimator_
        
        train_pred=reg_model.predict(x_train)
        test_pred=reg_model.predict(x_test)
        
        print(str(models).center(125,'='))
        print('Model:',reg_model)
        print('Train score:',r2_score(y_train,train_pred))
        print('Test score:',r2_score(y_test,test_pred))
        
        val_score=r2_score(y_test,test_pred)
        
        if val_score> best_score:
            best_score=val_score
            best_model=reg_model
            save_object(file_path='./artifacts',file_name='model.pkl',file_obj=best_model)

    print(f"Best Model: {best_model}")
    print(f"Validation Accuracy Score: {best_score}")


# model_train(models=model,params=params)