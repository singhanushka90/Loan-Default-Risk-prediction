import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import os


log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_data(data_url:str)->pd.DataFrame:
    """Load Data from a CSV file"""
    try:
        df=pd.read_csv(data_url)
        logger.debug("Data Loaded from %s",data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s',e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the data: %s",e)
        raise


def save_train_test_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    """Save train and test data"""
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug("Train and test data saved to %s",raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occured while saving the data: %s",e)
        raise


def main():
    try:
        test_size = 0.2
        data_path='experiments/application_train.csv'
        df=load_data(data_url=data_path)
        train_data,test_data=train_test_split(df,test_size=test_size,random_state=42,stratify=df["TARGET"])
        save_train_test_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logger.exception('Failed to complete the data ingestion process')
        raise

if __name__=='__main__':
    main()







