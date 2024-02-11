import hydra
from hydra import utils
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    f1_scoree = f1_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    return accuracy, f1_scoree, precision, recall


def model_building(X_train,X_test,Y_train,Y_test,model):
    lr = model.fit(X_train, Y_train)
    predicted_qualities = lr.predict(X_train)
    (accuracy, f1_scoree, precision, recall) = eval_metrics(Y_train, predicted_qualities)
    print("  accuracy: %s" % accuracy)
    print("  f1_scoree: %s" % f1_scoree)
    print("  precision: %s" % precision)
    print("  recall: %s" % recall)    

    return lr
  
        

@hydra.main(config_path='F:\\Machine_Learning_Ops\\mlops_1\\config',config_name='pre-processing',version_base="1.2")
def train_model(config):
    cwd1 = utils.get_original_cwd() + '\\'
    df_transform = pd.read_csv(cwd1 + config.processed.data_transformed)
    X = df_transform[config.features.input_features]
    Y = df_transform[config.features.target]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state= 45)
    model = RandomForestClassifier()
    final_model = model_building(X_train,X_test,Y_train,Y_test,model)
        # Save model
    #joblib.dump(final_model,config.model.path)
    with open(config.model.path, 'wb') as f:
        pickle.dump(final_model, f)
    

if __name__ == "__main__":
    train_model()
