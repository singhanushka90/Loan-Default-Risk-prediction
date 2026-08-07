import logging
import os
import pandas as pd
from config.features import SELECTED_FEATURES ,TARGET_COLUMN


log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_data_file=os.path.join(log_dir,'feature_engineering.log')
file_handler=logging.FileHandler(log_data_file)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_train_test_data(train_data:str,test_data:str):
    try:
        train=pd.read_csv(train_data)
        test=pd.read_csv(test_data)
        logger.debug("Load training data from: %s",train_data)
        logger.debug("Load testing data from: %s",test_data)
        return train , test
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv data: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error is occured during loading data: %s",e)
        raise


def select_features(train:pd.DataFrame,test:pd.DataFrame):
    try:
        train=train[SELECTED_FEATURES + TARGET_COLUMN]
        test=test[SELECTED_FEATURES + TARGET_COLUMN]
        logger.debug("Selected %d features successfully")
        len(SELECTED_FEATURES)
        return train , test
    except pd.errors.ParserError as e:
        logger.error("Error in parse data: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured during selection: %s",e)
        raise

def save_selected_data(train_df:pd.DataFrame,test_df:pd.DataFrame,data_path:str)->None:
    try:
        selected_data_path=os.path.join(data_path,'selected')
        os.makedirs(selected_data_path,exist_ok=True)
        train_df.to_csv(os.path.join(selected_data_path,'train.csv'),index=False)
        test_df.to_csv(os.path.join(selected_data_path,'test.csv'),index=False)
        logger.debug("Save Selected data Successfully")
    except Exception as e:
            logger.exception("Error wile saving processed data")
            raise


def main():
    try:
        train_df,test_df=load_train_test_data(train_data='data/raw/train.csv',test_data='data/raw/test.csv')
        train_df,test_df=select_features(train_df,test_df)
        save_selected_data(train_df=train_df,test_df=test_df,data_path="data")
        logger.info("Feature engineering completed successfully")
    except Exception as e:
        logger.exception("Feature engineering pipeline failed: %s",e)
        raise
if __name__=="__main__":
    main()

    
    