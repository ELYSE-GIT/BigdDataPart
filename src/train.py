import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import pandas as pd


reduced_trainData = pd.read_csv('data/data_pp.csv')

# split target and input variables
print('\separate TARGET from input model data ...\n')
X = reduced_trainData.iloc[:,1:].values
y = reduced_trainData['TARGET'].values

# split train and test data
print('\split train and test data ...\n')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=22)

# scale data
print('\nNormalize data ...\n')
min_max_scaler = MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)


print('Training lightgbm classifier model ...........')
# initialize
model = lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=22)

# train
model.fit(X_train_scaled, y_train, eval_metric='auc', 
          eval_set=[(X_train_scaled, y_train),(X_test_scaled, y_test)])

print('Saving model ...........')

from joblib import dump
dump(min_max_scaler, 'model/minMax_scaler_credits.joblib')
dump(model,'model/lgb_credits.joblib')

print('\nthe model importance :', model.feature_importances_,
      '\nvariables shape X :', np.shape(X),
      '\ncolumns with the target', reduced_trainData.columns) 


