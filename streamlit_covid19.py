import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from keras.models import load_model

import numpy as np
import pandas as pd

import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


api_url = 'https://api.coronavirus.data.gov.uk/v1/data'
response = requests.get(api_url)
data=response.json()
data

data1=data['data']
data1

import pandas as pd
df = pd.DataFrame(data1,columns=['date', 'areaName', 'areaCode', 'confirmedRate','latestBy','confirmed','deathNew','death','deathRate'])
print(df.head())

df.isnull().sum()

df.info()

df.describe()

df['confirmedRate']=df['confirmedRate'].fillna(df['confirmedRate'].median())

df['deathNew']=df['deathNew'].fillna(df['deathNew'].median())

df['death']=df['death'].fillna(df['death'].median())

df['deathRate']=df['deathRate'].fillna(df['deathRate'].median())
df.isnull().sum()

df.drop_duplicates(keep = 'first', inplace=True)

df.head()

df=df.drop(['date','areaName','areaCode'],axis=1)

x=df.loc[:,['confirmedRate','confirmed','deathNew','death','deathRate']]
y=df['latestBy']# latestBy =new cases
y

X2 = sm.add_constant(x)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

#conclusion1: d apres les resultas du tableau  OLS regression, on va garder x1,x2,x3 car pour alpha =0.05 on a p < 0.05 ( c est le cas de rejet de  l hypothese H0)

#etude de la correlation
corr = df.corr(method='spearman').round(2)
plt.figure(figsize = (25,20))
sns.heatmap(corr, annot = True, cmap = 'YlOrBr')

import matplotlib.pyplot as plt
plt.scatter(x=df['latestBy'],y=df['deathNew'])
plt.xlabel("new cases infectet")
plt.ylabel(' new deaths')
plt.title('new cases infectet vs new deaths')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(x=df['latestBy'],y=df['confirmed'])
plt.xlabel("new cases infectet")
plt.ylabel(' confirmed')
plt.title('new cases infectet vs confirmed')
plt.show()

from sklearn.preprocessing import StandardScaler
x=df.loc[:,['confirmedRate','confirmed','deathNew']]

scaler = StandardScaler()
x = scaler.fit_transform(x)
x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

y_t_pred=model.predict(x_test)
y_t_pred

from matplotlib.projections.polar import math
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn import metrics

#metric performance
#score_t=poly.score(x_test, y_t_pred)
MSE_t=mean_squared_error(y_test,y_t_pred)
RMSE_t=math.sqrt(MSE_t)
MAE_t=mean_absolute_error(y_test,y_t_pred)
#print('coefficient de determination : ',score_t)
print('mean square error :',MSE_t)
print('root mean square error :',RMSE_t)
print('mean absolute square error :',MAE_t)

print("R squared: ", metrics.r2_score(y_test,y_t_pred))

# cross validation 
from sklearn.model_selection import KFold
import numpy as np

X=pd.DataFrame(x)

# Define the number of folds for cross-validation
num_folds = 5

# Initialize variables for tracking the best model

# Create the cross-validation object
kf = KFold(n_splits=num_folds, shuffle=True)
RMSE_val_sc = []
RMSE_train_sc = []
R_squared_val_sc=[]
R_squared_train_sc=[]
# Perform cross-validation
for train_index, val_index in kf.split(X):
    x_train, x_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Create and train the model
    model = LinearRegression()

    model.fit(x_train, y_train)


    # Make predictions on the validation set
    y_pred_lr = model.predict(x_val)
    y_pred_lr_tr = model.predict(x_train)

    # Calculate the accuracy score for this
    #MSE=mean_squared_error(y_test,y_pred_lr)
    #RMSE=math.sqrt(MSE_t)
    #MAE=mean_absolute_error(y_test,y_t_pred)



    RMSE_val=mean_squared_error(y_val,y_pred_lr)
    RMSE_val_sc.append(RMSE_val)
    #print('RMSE in val set',RMSE_val_sc)
    RMSE_train=mean_squared_error(y_train,y_pred_lr_tr)
    RMSE_train_sc.append(RMSE_train)
    #print('RMSE in train set',RMSE_train_sc)

    R_squared_val=metrics.r2_score(y_val,y_pred_lr)
    R_squared_val_sc.append(R_squared_val)
    #print('R squared in val set',R_squared_val_sc)
    R_squared_train=metrics.r2_score(y_train,y_pred_lr_tr)
    R_squared_train_sc.append(R_squared_train)
    #print('R squared in train set',R_squared_train_sc)

best_RMSE_val =min(RMSE_val_sc)
avg_RMSE_val = sum(RMSE_val_sc)/num_folds
best_RMSE_train =min(RMSE_train_sc)
avg_RMSE_train = sum(RMSE_train_sc)/num_folds

best_R_squared_val =max(R_squared_val_sc)
avg_R_squared_val = sum(R_squared_val_sc)/num_folds
best_R_squared_train =max(R_squared_train_sc)
avg_R_squared_train = sum(R_squared_train_sc)/num_folds

print('best RMSE  is' ,{best_RMSE_val} , 'and  best R_suared  is ', {best_R_squared_val} ,'in val set')
print('Avg RMSE is' ,{avg_RMSE_val},' and average R_suared is', {avg_R_squared_val},' in val set')
print('best RMSE is',{best_RMSE_train},' and best R_suared is ', {best_R_squared_train},'in train set')
print('Avg RMSE is ', {avg_RMSE_train},' and average R_suared is', {avg_R_squared_train },'in train set')

# save the model to disk
#filename = 'finalized_model_regressionLinear.sav'
#pickle.dump(model, open(filename, 'wb'))

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=5)
poly.fit(x_train, y_train)
x_train_fit = poly.fit_transform(x_train) #transforming our input data
model.fit(x_train_fit, y_train)
x_test_ = poly.fit_transform(x_test)
predicted = model.predict(x_test_)
print('les metriques pour un modele  de regression Polynomiale d ordre 5')
print("MSE: ", metrics.mean_squared_error(y_test, predicted))
print("R squared: ", metrics.r2_score(y_test,predicted))





def predict_malade():
    st.title(' prediction of New cases infected WITH Covid19 ')
    
    # Input form
    st.subheader('Global Information')
    patient = {}
    patient['confirmedRate'] = st.number_input('donner le nombres des cas confirmes')
    patient['confirmed'] = st.number_input('donner le nombre des cas confirmes')
    patient['deathNew'] = st.number_input('donner le nombre des cas nouveau morts')
# Preprocess input data
    patient_df = pd.DataFrame(patient, index=[0])
    
    # Make prediction
    if st.button('Predict New cases infected ON Covid19'):
            poly=PolynomialFeatures(degree=5)
            #poly.fit(x_train, y_train)
            #x_train_fit = poly.fit_transform(x_train) #transforming our input data
            #model.fit(x_train_fit, y_train)
            x_test_ = poly.fit_transform(patient_df)

            #loaded_model = pickle.load(open(r'C:\Users\HP\Desktop\finalized_model_Polynomial_covid19.sav','rb'))
            prediction = model.predict(x_test_)
    
            
            st.write('number of new cases infected with Covid19 is :',prediction[0])
if __name__ == '__main__':
    predict_malade()
                   
