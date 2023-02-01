from joblib import load
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# loading models
min_max_scaler = load('model/minMax_scaler_credits.joblib')
model = load('model/lgb_credits.joblib')



reduced_trainData = pd.read_csv('data/data_pp.csv')

# split target and input variables
print('\separate TARGET from input model data ...\n')
X = reduced_trainData.iloc[:,1:].values
y = reduced_trainData['TARGET'].values

# scale data
print('\nNormalize data ...\n')
X_train_scaled = min_max_scaler.fit_transform(X)

print('\nprediction ...\n')
prediction = model.predict(X_train_scaled)

# accuracy
score = accuracy_score(y, prediction)
print('\nThe score : ', score)



# Constructing the confusion matrix based on train data
cm = confusion_matrix(prediction, y )

# Display the train confusion matrix
plt.figure(figsize=(6,6))
plt.title('Confusion matrix on train data')
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.plot
plt.show()