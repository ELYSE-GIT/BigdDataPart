import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import mlflow
import mlflow.lightgbm

reduced_trainData = pd.read_csv('data/data_pp.csv')

#split target and input variables
print('\separate TARGET from input model data ...\n')
X = reduced_trainData.iloc[:,1:].values
y = reduced_trainData['TARGET'].values

#split train and test data
print('\split train and test data ...\n')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=22)


#scale data
print('\nNormalize data ...\n')
min_max_scaler = MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)

#enable auto logging
mlflow.lightgbm.autolog()

with mlflow.start_run():
    print('Training lightgbm classifier model ...........')
    # initialize
    model = lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=22)

    # train
    model.fit(X_train_scaled, y_train, eval_metric='auc', 
            eval_set=[(X_train_scaled, y_train),(X_test_scaled, y_test)])
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("random_state", 22)
    mlflow.log_metric("auc", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1]))
    mlflow.log_artifact("data/data_pp.csv")
    mlflow.lightgbm.log_model(model, "model")
    print("mlflow run created!")

    #evaluate
    predict_train = model.predict_proba(X_train_scaled)[:,1]
    predict_test = model.predict_proba(X_test_scaled)[:,1]

    print('\nTrain AUC:',roc_auc_score(y_train, predict_train))
    print('\nTest AUC:',roc_auc_score(y_test, predict_test))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, model.predict(X_test_scaled))
    sns.heatmap(cm,annot=True,fmt = "d")
    plt.show()