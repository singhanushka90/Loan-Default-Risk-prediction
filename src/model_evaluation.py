import logging
import os
import pandas as pd
import numpy as np
from sklearn.metrics import  accuracy_score,f1_score,roc_auc_score,precision_score,recall_score,average_precision_score,precision_recall_curve,roc_curve
import matplotlib.pyplot as plt 
import joblib
import json
import mlflow

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("model_evaluation")
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model_evaluation.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_model(data_path:str):
    try:
        model_path=os.path.join(data_path,'model')
        model=joblib.load(os.path.join(model_path,'best_model.pkl'))
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error("Model file not found: %s",e)
        raise
    except Exception as e:
        logger.exception("Error while loading model: %s",e)
        raise

def evaluate_model(model,X_test,y_test):
    try:
        y_pred=model.predict(X_test)
        y_prob=model.predict_proba(X_test)[:,1]
        accuracy=accuracy_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred,zero_division=0)
        precision=precision_score(y_test,y_pred,zero_division=0)
        recall=recall_score(y_test,y_pred,zero_division=0)
        roc_auc=roc_auc_score(y_test,y_prob)
        average_precision=average_precision_score(y_test,y_prob)
        metrics={
            "accuracy" : accuracy,
            "precision" : precision,
            "recall" : recall,
            "f1_score" : f1,
            "roc_auc" : roc_auc,
            "average_precision" : average_precision
        }
        mlflow.log_metrics(metrics)
        logger.info("Evaluation metrics logger to MLFlow successfully")
        logger.info("Model evaluation completed successfully")
        logger.info("Accuracy : %.4f",accuracy)
        logger.info("Precision : %.4f",precision)
        logger.info("Recall : %.4f",recall)
        logger.info("F1_Score : %.4f",f1)
        logger.info("ROC-AUC : %.4f",roc_auc)
        logger.info("Average Precision : %.4f",average_precision)
        return metrics
    except Exception as e:
        logger.exception("Error during model evaluation : %s",e)
        raise

def plot_roc_curve(model,X_test,y_test,data_path:str):
    try:
        y_prob=model.predict_proba(X_test)[:,1]
        fpr,tpr,_=roc_curve(y_test,y_prob)
        auc_score=roc_auc_score(y_test,y_prob)
        plt.figure(figsize=(8,6))
        plt.plot(fpr,tpr,label=f"XGBoost (AUC ={auc_score:.3f})")
        plt.plot([0,1],[0,1],linestyle="--",label="XGB Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()
        plot_path=os.path.join(data_path,'plots')
        os.makedirs(plot_path,exist_ok=True)
        plt.savefig(os.path.join(plot_path,"roc_curve.png"),dpi=300,bbox_inches='tight')
        plt.show()
        plt.close()
        logger.info("ROC curve saved successfully")
        logger.info("ROC-AUC Score: %.4f",auc_score)
        return auc_score
    except Exception as e:
        logger.exception("Error while plotting ROC Curve: %s",e)
        raise


def plot_precision_recall_curve(model,X_test,y_test,data_path:str):
    try:
        y_prob=model.predict_proba(X_test)[:,1]
        precision,recall,_=precision_recall_curve(y_test,y_prob)
        ap_score=average_precision_score(y_test,y_prob)
        plt.figure(figsize=(8,6))
        plt.plot(recall,precision,label=f"XGBoost (AP ={ap_score:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision Recall Curve")
        plt.legend()
        plt.grid()
        plot_path=os.path.join(data_path,'plots')
        os.makedirs(plot_path,exist_ok=True)
        plt.savefig(os.path.join(plot_path,"Precision_Recall_Curve.png"),dpi=300,bbox_inches='tight')
        plt.show()
        plt.close()
        logger.info("Precision Recall curve saved successfully")
        logger.info("Average Precision: %.4f",ap_score)
        return ap_score
    except Exception as e:
        logger.exception("Error while plotting Precision Recall Curve: %s",e)
        raise

def save_metrics(metrics:dict,data_path:str):
    try:
        metrics_path=os.path.join(data_path,'metrics')
        os.makedirs(metrics_path,exist_ok=True)
        metrics_file=os.path.join(metrics_path,"metrics.json")
        with open(metrics_file,"w") as f:
            json.dump(metrics,f,indent=4)
        logger.info("Evaluation metrics saved successfully to: %s",metrics_file)
    except Exception as e:
        logger.exception("Error while saving metrics: %s",e)
        raise


def main():
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Loan Default Prediction")
        with mlflow.start_run(run_name="Model_Evaluation"):
            model=load_model(data_path="data")
            X_test=joblib.load(os.path.join("data","processed","X_test.pkl"))
            y_test=joblib.load(os.path.join("data","processed","y_test.pkl"))
            logger.info("Test data loaded successfully")
            metrics=evaluate_model(model=model,X_test=X_test,y_test=y_test)
            mlflow.log_metrics(metrics)
            logger.info("Evaluation metrics logger to MLFlow successfully")
            plot_roc_curve(model=model,X_test=X_test,y_test=y_test,data_path="data")
            plot_precision_recall_curve(model=model,X_test=X_test,y_test=y_test,data_path="data")
            save_metrics(metrics=metrics,data_path="data")
            logger.info("Model evaluation pipeline completed successfuly")
    except Exception as e:
        logger.exception("Model evaluation pipeline failed: %s",e)
        raise
if __name__=="__main__":
    main()


