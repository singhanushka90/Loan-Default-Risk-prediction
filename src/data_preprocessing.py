import os
import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from config.features import TARGET_COLUMN
import joblib

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_selected_data(train_data:str,test_data:str):
    """Load selected data"""
    try:
        train=pd.read_csv(train_data)
        test=pd.read_csv(test_data)
        logger.debug('Train and Test data loaded from: %s',train_data)
        logger.debug('Train and Test data loaded from: %s',test_data)
        return train , test
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data: %s',e)
        raise


def clean_data(df):
    try:
        df=df.copy()
        df["DAYS_EMPLOYED"]=df["DAYS_EMPLOYED"].replace(365243,np.nan)
        logger.debug('Clean data Successfully')
        return df 
    except KeyError as e:
        logger.error('Column not found: %s',e)
        raise
    except Exception as e:
        logger.error("Error during cleaning data: %s",e)
        raise


def split_features_target(df):
    try:
        X=df.drop(TARGET_COLUMN,axis=1)
        y=df[TARGET_COLUMN]
        logger.debug("Successfully split")
        return X,y
    except Exception as e:
        logger.error('Error during spliting the data: %s',e)
        raise



def create_preprocessor(X_train):
    try:
        categorical=X_train.select_dtypes(include=['object']).columns
        numerical=X_train.select_dtypes(exclude=['object']).columns
        logger.debug("categorical and numerical columns identified successfully")

        numerical_trans=Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
        categorical_trans=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(handle_unknown='ignore'))])
        logger.debug("Ready Pipeline Successfully")

        preprocessor=ColumnTransformer(transformers=[("num",numerical_trans,numerical),("cat",categorical_trans,categorical)])
        logger.debug("Preprocessing pipeline created successfully")
        return preprocessor
    except KeyError as e:
        logger.error('Column not found: %s',e)
        raise
    except Exception as e:
        
        logger.error("Error during preprocessor: %s",e)
        raise

def preprocess_data(X_train,X_test,preprocessor):
    try:
        X_train_processed=preprocessor.fit_transform(X_train)
        X_test_processed=preprocessor.transform(X_test)
        logger.info("Data Preprocessing completed successfullly")
        return X_train_processed,X_test_processed
    except Exception as e:
        logger.exception("Error during Preprocessing: %s",e)
        raise


def save_processed_data(X_train_processed,X_test_processed,y_train,y_test,preprocessor,data_path:str)->None:
    try:
        processed_data_path=os.path.join(data_path,'processed')
        os.makedirs(processed_data_path,exist_ok=True)
        joblib.dump(X_train_processed,os.path.join(processed_data_path,'X_train.pkl'))
        joblib.dump(X_test_processed,os.path.join(processed_data_path,'X_test.pkl'))
        joblib.dump(y_train,os.path.join(processed_data_path,'y_train.pkl'))
        joblib.dump(y_test,os.path.join(processed_data_path,'y_test.pkl'))
        joblib.dump(preprocessor,os.path.join(processed_data_path,'preprocessor.pkl'))
        logger.info("Processed data saved successfully")
    except Exception as e:
        logger.exception("Error while saving processed data: %s",e)
        raise

    
def main():
    try:
        train_df,test_df=load_selected_data(train_data="data/selected/train.csv",test_data="data/selected/test.csv")

        train_df=clean_data(train_df)
        test_df=clean_data(test_df)

        X_train,y_train=split_features_target(train_df)
        X_test,y_test=split_features_target(test_df)

        preprocessor=create_preprocessor(X_train)

        X_train_processed,X_test_processed=preprocess_data(X_train,X_test,preprocessor)

        save_processed_data(X_train_processed=X_train_processed,X_test_processed=X_test_processed,y_train=y_train,y_test=y_test,preprocessor=preprocessor,data_path="data")

        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        logger.exception("Failed to complete data preprocessing: %s",e)
        raise
if __name__=="__main__":
    main()









