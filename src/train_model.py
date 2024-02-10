import hydra
from hydra import utils
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import warnings
import sys
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    f1_scoree = f1_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    return accuracy, f1_scoree, precision, recall


def model_building(X_train,X_test,Y_train,Y_test,model,exp_name,model_name,run_name):
    filter_string = f"name = '{exp_name}'"
    results  = mlflow.search_experiments(filter_string= filter_string)
    name = [res.name for res in results]
    print(name)
    if exp_name not in name:
            mlflow.create_experiment(exp_name)

    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=run_name):
        lr = model.fit(X_train, Y_train)
        predicted_qualities = lr.predict(X_test)
        (accuracy, f1_scoree, precision, recall) = eval_metrics(Y_test, predicted_qualities)
                
        print("  accuracy: %s" % accuracy)
        print("  f1_scoree: %s" % f1_scoree)
        print("  precision: %s" % precision)
        print("  recall: %s" % recall)    
        fpr, tpr, _ = roc_curve(Y_test, predicted_qualities)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
                # Save ROC curve plot to a file
        roc_curve_image_path = "F:\\Machine_Learning_Ops\\mlops_1\\graphs\\" + model_name + ".png"
        plt.savefig(roc_curve_image_path)
        plt.close()
        mlflow.log_artifact(roc_curve_image_path, artifact_path="roc_curve_images")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_scoree", f1_scoree)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)     
 
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name= model_name)
        else:
            mlflow.sklearn.log_model(lr, "model")   


@hydra.main(config_path='F:\\Machine_Learning_Ops\\mlops_1\\config',config_name='pre-processing',version_base="1.2")
def train_model(config):
    cwd1 = utils.get_original_cwd() + '\\'
    df_transform = pd.read_csv(cwd1 + config.processed.data_transformed)
    X = df_transform[config.features.input_features]
    Y = df_transform[config.features.target]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
    model = RandomForestClassifier()
    exp_name = 'Employee Churn'
    model_name = 'RandomForestClassifier'
    run_name = 'RandomForestClassifier'
    model_building(X_train,X_test,Y_train,Y_test,model,exp_name,model_name,run_name)
    

if __name__ == "__main__":
    train_model()
