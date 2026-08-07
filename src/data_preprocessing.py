import os
import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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


def load_train_test_data(train_data:str,test_data:str):
    """Load train and test data"""
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
        X=df.drop('TARGET',axis=1)
        y=df['TARGET']
        logger.debug("Successfully split")
        return X,y
    except Exception as e:
        logger.error('Error during spliting the data: %s',e)



def create_preprocesssor(X_train):
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


def preprocessor_data(X_train,X_test,preprocessor):
    try:
        X_train_processed=preprocessor.fit_transform(X_train)
        X_test_processed=preprocessor.transform(X_test)
        logger.debug("Preprocessing completed successfullly")
        return X_train_processed,X_test_processed,preprocessor
    except Exception as e:
        logger.exception("Error during Preprocessing: %s",e)
        raise


def save_processed_data(X_train_processed,X_test_processed,y_train,y_test,data_path:str):
    try:
        processed_data=os.path.join(data_path,"processed")
        os.makedirs(processed_data,exist_ok=True)
        X_train_processed=pd.DataFrame(X_train_processed)
        X_test_processed=pd.DataFrame(X_test_processed)
        X_train_processed["TARGET"]=y_train.reset_index(drop=True)
        X_test_processed["TARGET"]=y_test.reset_index(drop=True)
        X_train_processed.to_csv(os.path.join(processed_data,"processed_train.csv"),index=False)
        X_test_processed.to_csv(os.path.join(processed_data,"processed_test.csv"),index=False)
        logger.info("Processed data saved successfully")

    except Exception as e:
        logger.exception("Error wile saving processed data")
        raise
    


def main():
    try:
        train_df,test_df=load_train_test_data(train_data="data/raw/train.csv",test_data="data/raw/test.csv")

        train_df=clean_data(train_df)
        test_df=clean_data(test_df)

        X_train,y_train=split_features_target(train_df)
        X_test,y_test=split_features_target(test_df)

        preprocessor=create_preprocesssor(X_train)

        X_train_processed,X_test_processed,preprocessor=preprocessor_data(X_train,X_test,preprocessor)

        save_processed_data(X_train_processed,X_test_processed,y_train,y_test,data_path="data")

        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        logger.exception("Failed to complete data preprocessing")
        raise
if __name__=="__main__":
    main()
