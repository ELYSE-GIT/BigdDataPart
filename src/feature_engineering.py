import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import pandas as pd


# load dataset
#trainData = pd.read_csv('data/application_test.csv')
trainData = pd.read_csv('data/application_train.csv')

# display
print('shape of the dataset:', trainData.shape,
    '\ncolumns in the dataset:', trainData.columns)

# a function to fix age representation
def convert_age(age_days_negative):
    age_days_positive = -age_days_negative
    age_years = age_days_positive/365
    return age_years

trainData['DAYS_BIRTH'] = trainData['DAYS_BIRTH'].apply(convert_age)
trainData['DAYS_EMPLOYED'] = trainData['DAYS_EMPLOYED'].apply(convert_age)


used_features = [
    'TARGET',
    'NAME_CONTRACT_TYPE',
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'CNT_CHILDREN',
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_GOODS_PRICE',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'CNT_FAM_MEMBERS',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3'
]

# Create a new data frame which only consists of those selected columns
reduced_trainData = trainData[used_features]

print('\nfeatures types kept \n',reduced_trainData.dtypes)

print('\nshape of preprocessed data before encoding:', reduced_trainData.shape )
print('\ncolumns of preprocessed data before encoding: ', reduced_trainData.columns)
print('\n')
print(reduced_trainData.head())

print('\ndifferent contents for each columns\n')
for column in reduced_trainData.columns:
    print("{}\t: {}".format(column, len(np.unique(reduced_trainData[column]))))



columns = ['CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']

print ('\napply one hot encoder on :', columns)
def create_one_hot(reduced_trainData, columns):
    for column in columns:
        reduced_trainData = pd.concat([reduced_trainData, pd.get_dummies(trainData[column])], axis=1, join='inner')
        reduced_trainData = reduced_trainData.drop([column], axis=1)
    
    return reduced_trainData

reduced_trainData = create_one_hot(reduced_trainData, columns)


print ('\napply label encoder on :', [ 'NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY' ])
le_name_contract_type = LabelEncoder()
reduced_trainData['NAME_CONTRACT_TYPE'] = le_name_contract_type.fit_transform(reduced_trainData['NAME_CONTRACT_TYPE'])

le_flag_own_car = LabelEncoder()
reduced_trainData['FLAG_OWN_CAR'] = le_flag_own_car.fit_transform(reduced_trainData['FLAG_OWN_CAR'])

le_flag_own_realty = LabelEncoder()
reduced_trainData['FLAG_OWN_REALTY'] = le_flag_own_realty.fit_transform(reduced_trainData['FLAG_OWN_REALTY'])


# print variables with missing data only
print('\nThe variables with missing data :')

for i in range (reduced_trainData.shape[1]):
    if reduced_trainData.isnull().sum()[i] !=0 :
        print(reduced_trainData.isnull().sum().index[i]) 

print('\nimputation of these data with mean method ...')
reduced_trainData.loc[:,'AMT_GOODS_PRICE'] = reduced_trainData['AMT_GOODS_PRICE'].fillna(reduced_trainData['AMT_GOODS_PRICE'].mean())
reduced_trainData.loc[:,'CNT_FAM_MEMBERS'] = reduced_trainData['CNT_FAM_MEMBERS'].fillna(reduced_trainData['CNT_FAM_MEMBERS'].mean())
reduced_trainData.loc[:,'EXT_SOURCE_1'] = reduced_trainData['EXT_SOURCE_1'].fillna(reduced_trainData['EXT_SOURCE_1'].mean())
reduced_trainData.loc[:,'EXT_SOURCE_2'] = reduced_trainData['EXT_SOURCE_2'].fillna(reduced_trainData['EXT_SOURCE_2'].mean())
reduced_trainData.loc[:,'EXT_SOURCE_3'] = reduced_trainData['EXT_SOURCE_3'].fillna(reduced_trainData['EXT_SOURCE_3'].mean())

print('\nshape of preprocessed data :', reduced_trainData.shape )
print('\ncolumns of preprocessed data : ', reduced_trainData.columns)
print('\n')
print(reduced_trainData.head())

print('\nexporting data in ../src/data \n')
print('*********************************************************************************')
#### Export preprocessed data 
reduced_trainData.to_csv('../src/data/data_ppt.csv', index=False)