# Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import pandas as pd
import shap
import joblib
from joblib import dump

# Load the data
reduced_trainData = pd.read_csv('data/data_pp.csv')

# Split the target and input variables
X = reduced_trainData.iloc[:,1:].values
y = reduced_trainData['TARGET'].values

# Split the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=22)

# Scale the data
min_max_scaler = MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)

# Train the LightGBM model
model = lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=22)
model.fit(X_train_scaled, y_train, eval_metric='auc', eval_set=[(X_train_scaled, y_train),(X_test_scaled, y_test)])

# Save the model and scaler
dump(min_max_scaler, 'model/minMax_scaler_credits.joblib')
dump(model,'model/lgb_credits.joblib')

# Load the saved model and scaler
min_max_scaler = joblib.load('model/minMax_scaler_credits.joblib')
model = joblib.load('model/lgb_credits.joblib')

# Build a TreeExplainer and compute Shapley Values
explainer = shap.Explainer(model.predict_proba, X_train_scaled)
shap_values = explainer(X_test_scaled)

# Visualize explanations for a specific point of the data set
index = 10
shap.plots.waterfall(shap_values[index], X_test_scaled[index])
plt.title(f"SHAP Explanations for Test Point {index}")

# Visualize explanations for all points of the data set at once
shap.plots.waterfall(shap_values, X_test_scaled)
plt.title("SHAP Explanations for All Test Points")

# Visualize a summary plot for each class on the whole dataset
shap.plots.summary_plot(shap_values, X_test_scaled)
plt.title("SHAP Summary Plot")
