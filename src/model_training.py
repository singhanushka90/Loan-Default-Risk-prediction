import os
import logging
import pandas as pd 
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model_training.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_processed_data(data_path:str):
    try:
        processed_data_path=os.path.join(data_path,"processed")
        X_train=joblib.load(os.path.join(processed_data_path,"X_train.pkl"))
        X_test=joblib.load(os.path.join(processed_data_path,"X_test.pkl"))
        y_train=joblib.load(os.path.join(processed_data_path,"y_train.pkl"))
        y_test=joblib.load(os.path.join(processed_data_path,"y_test.pkl"))

        logger.info("Processed data loaded successfully")

        return X_train,X_test,y_train,y_test
    except FileNotFoundError as e:
        logger.error("Processed file not found: %s",e)
        raise
    except Exception as e:
        logger.exception("Error while loading data: %s",e)
        raise


def build_model():
    try:
        model=XGBClassifier(random_state=42,eval_metric='logloss',scale_pos_weight=11)
        logger.info("Model built successfully")
        return model
    except Exception as e:
        logger.exception("Error occured during building model: %s",e)
        raise

def train_model(model,X_train,y_train):
    try:
        model.fit(X_train,y_train)
        logger.info("Model trained Successfully")
        return model
    except Exception as e:
        logger.error("Error occured during training model: %s",e)
        raise

def hyperparameter_tuning(model,X_train,y_train):
    try:
        params_dist={
            "n_estimators":[100,200,300,500],
            "max_depth":[3,5,7,9],
            "learning_rate":[0.01,0.05,0.1,0.2],
            "subsample":[0.6,0.8,1.0],
            "colsample_bytree":[0.6,0.8,1.0],
            "scale_pos_weight":[1,3,5,11],
            "gamma":[0,0.1,0.3,0.5]
        }
        random_search=RandomizedSearchCV(estimator=model,param_distributions=params_dist,n_iter=20,cv=3,verbose=2,scoring="f1",random_state=42,n_jobs=-1,)
        random_search.fit(X_train,y_train)
        logger.info("Best Parameters: %s",random_search.best_params_)
        logger.info("Best Score : %s",random_search.best_score_)
        logger.info("Successfully completed hyperparameter")
        return random_search.best_estimator_
    except Exception as e:
        logger.error("Error occured during hyperparameter tuning: %s",e)
        raise

def save_model(best_model,data_path):
    try:
        model_path=os.path.join(data_path,"model")
        os.makedirs(model_path,exist_ok=True)
        joblib.dump(best_model,os.path.join(model_path,'best_model.pkl'))
        logger.info("Successfully save model")
    except Exception as e:
            logger.exception("Error while saving processed data: %s",e)
            raise

def main():
    try:
        X_train,X_test,y_train,y_test,=load_processed_data(data_path="data")
        model=build_model()
        train_model(model,X_train,y_train)
        best_model=hyperparameter_tuning(model,X_train,y_train)
        save_model(best_model=best_model,data_path="data")
        logger.info("Model training pipeline completed successfully")
    except Exception as e:
        logger.exception("Model training pipeline failed: %s",e)
        raise

if __name__=='__main__':
    main()


