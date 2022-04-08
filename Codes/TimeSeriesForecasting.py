# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:11:07 2022

@author: Yikes and Rohit
"""

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import datetime
import sklearn
from sklearn import *
sklearn.__version__
import numpy as np 
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from time import time
import itertools
from scipy.stats.stats import mode
from statsmodels.tsa.statespace import sarimax
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from fbprophet import Prophet

import os
os.chdir('..')

## Data Input

df_scaling = pd.read_csv("Data/Scaling.csv")
df_scaling_division=df_scaling[["Item_ID","Division_ID","Division_Scaling_Factor"]]
df_scaling_cluster=df_scaling[["Item_ID","Cluster_ID","Cluster_Scaling_Factor"]]

df_items= pd.read_csv("Data/Item.csv",parse_dates=['Ship_Date'])
df_division = pd.read_csv("Data/Division.csv", parse_dates=['Ship_Date'])
df_cluster = pd.read_csv("Data/Cluster.csv", parse_dates=['Ship_Date'])
result_rf = pd.read_csv("Data/RandomForest_Result.csv")
df_rf = pd.read_csv("Data/RandomForest_Out.csv", parse_dates=['Ship_Date'])


divisions=list(df_division["Division_ID"].unique())
clusters=list(df_cluster["Cluster_ID"].unique())

#Holt Winter Forecasting - Returns training and testing datasets with predicted values 
def Holt_Winter_Forecasting(training_dataset,testing_dataset,operation):
  training_dataset.head(5)
  training_dataset=training_dataset.set_index("Ship_Date")
  testing_dataset=testing_dataset.set_index("Ship_Date")
  y_train=training_dataset["Ship_Qty"]
  fitted_model = ExponentialSmoothing(y_train,trend=operation,seasonal=operation,seasonal_periods=13).fit(optimized=True)
  testing_dataset["Prediction_Holt"] = list(fitted_model.forecast(len(testing_dataset)))
  training_dataset["Prediction_Holt"]=list(fitted_model.forecast(len(training_dataset)))
  return training_dataset.reset_index(),testing_dataset.reset_index()


#RMSE AND MAPE
def calculate_rmse(actual,predicted):
  rmse = np.sqrt(mean_squared_error(actual, predicted)).round(2)
  return rmse
def calculate_mape(actual,predicted):
  mape= np.round(np.mean(np.abs(actual-predicted)/actual)*100,2)
  return mape


"""--------------------------Holt Winter---------------------------------"""

#Train-Test DataSet: 80% training 20% testing
training_division_dataset=df_division[df_division["Ship_Date"]<datetime.datetime(2019,11,1)]
testing_division_dataset=df_division[df_division["Ship_Date"]>=datetime.datetime(2019,11,1)]

#x_train= training_dataset[["Division","Ship_Date"]]
training_division_dataset=training_division_dataset.set_index("Ship_Date")
testing_division_dataset=testing_division_dataset.set_index("Ship_Date")
y_train=training_division_dataset["Ship_Qty"]

#x_test= testing_dataset[["Division","Ship_Date"]]
y_test=testing_division_dataset["Ship_Qty"]

#Forecasting
operation="add"
df_test=pd.DataFrame()
#df_holt=pd.DataFrame(columns=["Ship_Date","Division","Ship_Qty","Prediction"])
for div in divisions:
  training_dataset=training_division_dataset[training_division_dataset["Division_ID"]==div].reset_index()
  testing_dataset=testing_division_dataset[testing_division_dataset["Division_ID"]==div].reset_index()
  training_dataset,testing_dataset=Holt_Winter_Forecasting(training_dataset,testing_dataset,operation)
  df_test=df_test.append(testing_dataset)

#Plotting
df_test=df_test.set_index("Ship_Date")
plot_title="Holt_Winter_Forecast" + "_" + operation
y_train.plot(legend=True,label='TRAIN')
y_test.plot(legend=True,label='TEST',figsize=(6,4))
df_test["Prediction_Holt"].plot(legend=True,label='PREDICTION')
plt.title(plot_title)
   
#Merging
df_test=df_test.reset_index()
df_holt_division = pd.merge(df_test,df_scaling_division,how='left',on="Division_ID")
df_holt_division["Item_forecast_Holt"]=df_holt_division["Prediction_Holt"]*df_holt_division["Division_Scaling_Factor"]
df_holt_division=df_holt_division[["Ship_Date","Division_ID","Item_ID","Item_forecast_Holt"]]
df_holt_division=pd.merge(df_holt_division,df_items[["Item_ID","Ship_Date","Ship_Qty"]],on=["Item_ID","Ship_Date"],how="left")
#df_holt_division=df_holt_division[df_holt_division["Ship_Qty"]>0]


items=list(df_items["Item_ID"].unique())
result=pd.DataFrame()
result2=pd.DataFrame()
for item in items:
  df=df_holt_division[df_holt_division["Item_ID"]==item]
  actual=df["Ship_Qty"]
  predicted=df["Item_forecast_Holt"]
  mape_holt= calculate_mape(actual,predicted)
  rmse_holt=calculate_rmse(actual,predicted)
  holt_result=pd.DataFrame({'Method':["Holt Winter"],"Item_ID":[item],'RMSE':[rmse_holt],'MAPE':[mape_holt]})
  result=pd.concat([result,holt_result],axis=0)
  holt_result2= pd.DataFrame({"Item_ID":[item],'RMSE_Holt':[rmse_holt],'MAPE_Holt':[mape_holt]})
  result2=pd.concat([result2,holt_result2],axis=0)
  
  
  
"""----------------------------------SARIMA---------------------------"""


def SARIMAX_Forecasting(training_dataset,testing_dataset,order,seasonal_order):
  model = sarimax.SARIMAX(training_dataset["Ship_Qty"], order=order, seasonal_order=seasonal_order)
  model_fit=model.fit()
  testing_dataset["Prediction_Sarima"]= model_fit.predict(start=44,end=52).reset_index().drop("index",1)
  return testing_dataset

#Attributes selected by acf, pacf plots and Gridsearch
attribute_dict={"US-RTL":[[1,1,0],[1, 2, 0, 13]],"CA-FS":[[0,1,2],[1, 2, 0, 13]],"US-FS":[[1,1,0],[1, 2, 0, 13]],"CA-RTL":[[0,1,2],[1, 2, 0, 13]]}


df_test=pd.DataFrame()
for div in divisions:
  order=tuple(attribute_dict[div][0])
  seasonal_order=tuple(attribute_dict[div][1])
  training_dataset=training_division_dataset[training_division_dataset["Division_ID"]==div].reset_index()
  testing_dataset=testing_division_dataset[testing_division_dataset["Division_ID"]==div].reset_index()
  testing_dataset=SARIMAX_Forecasting(training_dataset,testing_dataset,order,seasonal_order)
  df_test=df_test.append(testing_dataset)
  
#Merging
df_sarima_division = pd.merge(df_test,df_scaling,how='left',on="Division_ID")
df_sarima_division["Item_forecast_Sarima"]=df_sarima_division["Prediction_Sarima"]*df_sarima_division["Division_Scaling_Factor"]
df_sarima_division=df_sarima_division[["Ship_Date","Division_ID","Item_ID","Item_forecast_Sarima"]]
df_sarima_division=pd.merge(df_sarima_division,df_items[["Item_ID","Ship_Date","Ship_Qty"]],on=["Item_ID","Ship_Date"],how="left")
#df_sarima_division=df_sarima_division[df_sarima_division["Ship_Qty"]>0]

sarima_result2=pd.DataFrame()
for item in items:
  df=df_sarima_division[df_sarima_division["Item_ID"]==item]
  actual=df["Ship_Qty"]
  predicted=df["Item_forecast_Sarima"]
  mape_sarima= calculate_mape(actual,predicted)
  rmse_sarima=calculate_rmse(actual,predicted)
  sarima_result=pd.DataFrame({'Method':["SARIMA"],"Item_ID":[item],'RMSE':[rmse_sarima],'MAPE':[mape_sarima]})
  result=pd.concat([result,sarima_result],axis=0)
  sarima_result2= pd.concat([sarima_result2,pd.DataFrame({"Item_ID":[item],'RMSE_SARIMA':[rmse_sarima],'MAPE_SARIMA':[mape_sarima]})],axis=0)
result2=pd.merge(result2,sarima_result2,on=["Item_ID"],how="left")

result=result.reset_index().drop("index",1)
df1=result.groupby(["Item_ID","Method"]).agg("max")


"""-------------------------Prophets-----------------------------------"""
def Prophet_Forecasting(training_dataset,testing_dataset):
  df_train = training_dataset.rename({'Ship_Date': 'ds', 'Ship_Qty': 'y'}, axis=1)
  df_test=pd.DataFrame()
  df_test["ds"]=testing_dataset["Ship_Date"]
  model=Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
  model.add_seasonality(name='quaterly', period=13, fourier_order=6)
  model.fit(df_train)
  testing_dataset["Prediction_Prophet"]=model.predict(df_test)["yhat"]
  return testing_dataset

df_test=pd.DataFrame()
for div in divisions:
  training_dataset=training_division_dataset[training_division_dataset["Division_ID"]==div].drop("Division_ID",1).reset_index()
  testing_dataset=testing_division_dataset[testing_division_dataset["Division_ID"]==div].reset_index()
  testing_dataset=Prophet_Forecasting(training_dataset,testing_dataset)
  df_test=df_test.append(testing_dataset)
  
  #Merging
df_prophet_division = pd.merge(df_test,df_scaling,how='left',on="Division_ID")
df_prophet_division["Item_Forecast_Prophet"]=df_prophet_division["Prediction_Prophet"]*df_prophet_division["Division_Scaling_Factor"]
df_prophet_division = df_prophet_division[["Ship_Date","Division_ID","Item_ID","Item_Forecast_Prophet"]]
df_prophet_division = pd.merge(df_prophet_division,df_items[["Item_ID","Ship_Date","Ship_Qty"]],on=["Item_ID","Ship_Date"],how="left")
#df_prophet_division = df_prophet_division[df_prophet_division["Ship_Qty"]>0]  #remove

#Results
prophet_result2=pd.DataFrame()
for item in items:
  df=df_prophet_division[df_prophet_division["Item_ID"]==item]
  actual=df["Ship_Qty"]
  predicted=df["Item_Forecast_Prophet"]
  mape_prophet= calculate_mape(actual,predicted)
  rmse_prophet=calculate_rmse(actual,predicted)
  prophet_result=pd.DataFrame({'Method':["Prophet"],"Item_ID":[item],'RMSE':[rmse_prophet],'MAPE':[mape_prophet]})
  result=pd.concat([result,prophet_result],axis=0)
  prophet_result2= pd.concat([prophet_result2,pd.DataFrame({"Item_ID":[item],'RMSE_Prophet':[rmse_prophet],'MAPE_Prophet':[mape_prophet]})],axis=0)
result2=pd.merge(result2,prophet_result2,on=["Item_ID"],how="left")

result=result.groupby(["Item_ID","Method"]).agg("max")


"""----------------------------------Cluster Level---------------------------------------"""
"""-----------------------------------HoltWinter-----------------------------------------"""

#Train-Test DataSet: 80% training 20% testing
training_cluster_dataset=df_cluster[df_cluster["Ship_Date"]<datetime.datetime(2019,11,1)]
testing_cluster_dataset=df_cluster[df_cluster["Ship_Date"]>=datetime.datetime(2019,11,1)]

#x_train= training_dataset[["Division","Ship_Date"]]
training_cluster_dataset=training_cluster_dataset.set_index("Ship_Date")
testing_cluster_dataset=testing_cluster_dataset.set_index("Ship_Date")
y_train=training_cluster_dataset["Ship_Qty"]

#x_test= testing_dataset[["Division","Ship_Date"]]
y_test=testing_cluster_dataset["Ship_Qty"]

#Forecasting
operation="add"
df_test=pd.DataFrame()
#df_holt=pd.DataFrame(columns=["Ship_Date","Division","Ship_Qty","Prediction"])
for cluster in clusters:
  training_dataset=training_cluster_dataset[training_cluster_dataset["Cluster_ID"]==cluster].reset_index()
  testing_dataset=testing_cluster_dataset[testing_cluster_dataset["Cluster_ID"]==cluster].reset_index()
  training_dataset,testing_dataset=Holt_Winter_Forecasting(training_dataset,testing_dataset,operation)
  df_test=df_test.append(testing_dataset)

#Plotting
df_test=df_test.set_index("Ship_Date")
plot_title="Holt_Winter_Forecast" + "_" + operation
y_train.plot(legend=True,label='TRAIN')
y_test.plot(legend=True,label='TEST',figsize=(6,4))
df_test["Prediction_Holt"].plot(legend=True,label='PREDICTION')
plt.title(plot_title)
   
#Merging
df_test=df_test.reset_index()
df_holt_cluster = pd.merge(df_test,df_scaling_cluster,how='left',on="Cluster_ID")
df_holt_cluster["Item_forecast_Holt"]=df_holt_cluster["Prediction_Holt"]*df_holt_cluster["Cluster_Scaling_Factor"]
df_holt_cluster=df_holt_cluster[["Ship_Date","Cluster_ID","Item_ID","Item_forecast_Holt"]]
df_holt_cluster=pd.merge(df_holt_cluster,df_items[["Item_ID","Ship_Date","Ship_Qty"]],on=["Item_ID","Ship_Date"],how="left")
#df_holt_cluster=df_holt_cluster[df_holt_cluster["Ship_Qty"]>0]


#Results
items=list(df_items["Item_ID"].unique())
result_cluster=pd.DataFrame()
result_cluster2=pd.DataFrame()
for item in items:
  df=df_holt_cluster[df_holt_cluster["Item_ID"]==item]
  actual=df["Ship_Qty"]
  predicted=df["Item_forecast_Holt"]
  mape_holt= calculate_mape(actual,predicted)
  rmse_holt=calculate_rmse(actual,predicted)
  holt_result=pd.DataFrame({'Method':["Holt Winter"],"Item_ID":[item],'RMSE':[rmse_holt],'MAPE':[mape_holt]})
  result_cluster=pd.concat([result_cluster,holt_result],axis=0)
  holt_result2= pd.DataFrame({"Item_ID":[item],'RMSE_Holt':[rmse_holt],'MAPE_Holt':[mape_holt]})
  result_cluster2=pd.concat([result_cluster2,holt_result2],axis=0)
#result_cluster2=pd.merge(result_cluster2,df_random_cluster,on=["Item_ID"],how="left")


"""----------------------------------------SARIMA---------------------------"""
#Attributes selected by acf, pacf plots and Gridsearch
attribute_cluster_dict={0:[[0,1,0],[0, 2, 0, 13]],1:[[0,1,1],[1,2,0,13]],2:[[1,1,0],[0,2,0,13]],3:[[1,1,0],[1, 2, 0, 13]]}


df_test=pd.DataFrame()
for cluster in clusters:
  order=(1,1,0)
  seasonal_order=(3,2,1,8)
  training_dataset=training_cluster_dataset[training_cluster_dataset["Cluster_ID"]==cluster].reset_index()
  testing_dataset=testing_cluster_dataset[testing_cluster_dataset["Cluster_ID"]==cluster].reset_index()
  testing_dataset=SARIMAX_Forecasting(training_dataset,testing_dataset,order,seasonal_order)
  df_test=df_test.append(testing_dataset)
  
  #Merging
df_sarima_cluster = pd.merge(df_test,df_scaling_cluster,how='left',on="Cluster_ID")
df_sarima_cluster["Item_forecast_Sarima"]=df_sarima_cluster["Prediction_Sarima"]*df_sarima_cluster["Cluster_Scaling_Factor"]
df_sarima_cluster=df_sarima_cluster[["Ship_Date","Cluster_ID","Item_ID","Item_forecast_Sarima"]]
df_sarima_cluster=pd.merge(df_sarima_cluster,df_items[["Item_ID","Ship_Date","Ship_Qty"]],on=["Item_ID","Ship_Date"],how="left")
#df_sarima_cluster=df_sarima_cluster[df_sarima_cluster["Ship_Qty"]>0]
#Results
sarima_result2=pd.DataFrame()
for item in items:
  df=df_sarima_cluster[df_sarima_cluster["Item_ID"]==item]
  actual=df["Ship_Qty"]
  predicted=df["Item_forecast_Sarima"]
  mape_sarima= calculate_mape(actual,predicted)
  rmse_sarima=calculate_rmse(actual,predicted)
  sarima_result=pd.DataFrame({'Method':["SARIMA"],"Item_ID":[item],'RMSE':[rmse_sarima],'MAPE':[mape_sarima]})
  result_cluster=pd.concat([result_cluster,sarima_result],axis=0)
  sarima_result2= pd.concat([sarima_result2,pd.DataFrame({"Item_ID":[item],'RMSE_SARIMA':[rmse_sarima],'MAPE_SARIMA':[mape_sarima]})],axis=0)
result_cluster2=pd.merge(result_cluster2,sarima_result2,on=["Item_ID"],how="left")

"""------------------------------Propehts-------------------------------------"""
df_test=pd.DataFrame()
for cluster in clusters:
  training_dataset=training_cluster_dataset[training_cluster_dataset["Cluster_ID"]==cluster].drop("Cluster_ID",1).reset_index()
  testing_dataset=testing_cluster_dataset[testing_cluster_dataset["Cluster_ID"]==cluster].reset_index()
  testing_dataset=Prophet_Forecasting(training_dataset,testing_dataset)
  df_test=df_test.append(testing_dataset)
#Merging
df_prophet_cluster = pd.merge(df_test,df_scaling,how='left',on="Cluster_ID")
df_prophet_cluster["Item_Forecast_Prophet"] = df_prophet_cluster["Prediction_Prophet"]*df_prophet_cluster["Cluster_Scaling_Factor"]
df_prophet_cluster = df_prophet_cluster[["Ship_Date","Cluster_ID","Item_ID","Item_Forecast_Prophet"]]
df_prophet_cluster = pd.merge(df_prophet_cluster,df_items[["Item_ID","Ship_Date","Ship_Qty"]],on=["Item_ID","Ship_Date"],how="left")
#df_prophet_cluster=df_prophet_cluster[df_prophet_cluster["Ship_Qty"]>0]  #remove

#Results
prophet_result2=pd.DataFrame()
for item in items:
  df=df_prophet_cluster[df_prophet_cluster["Item_ID"]==item]
  actual=df["Ship_Qty"]
  predicted=df["Item_Forecast_Prophet"]
  mape_prophet= calculate_mape(actual,predicted)
  rmse_prophet=calculate_rmse(actual,predicted)
  prophet_result=pd.DataFrame({'Method':["Prophet"],"Item_ID":[item],'RMSE':[rmse_prophet],'MAPE':[mape_prophet]})
  result_cluster=pd.concat([result_cluster,prophet_result],axis=0)
  prophet_result2= pd.concat([prophet_result2,pd.DataFrame({"Item_ID":[item],'RMSE_Prophet':[rmse_prophet],'MAPE_Prophet':[mape_prophet]})],axis=0)
result_cluster2=pd.merge(result_cluster2,prophet_result2,on=["Item_ID"],how="left")


#------------------------------ Karan 

result = result.reset_index().rename(columns= {"Method":"Forecast"})
result_cluster = result_cluster.rename(columns = {"Method": "Forecast"})

result["Agg_Level"] = "Division"
result_cluster["Agg_Level"] = "Cluster"

result_final = pd.concat((result_rf, result), axis=0)
result_final = pd.concat((result_final, result_cluster), axis=0).reset_index().drop("index", axis=1)

best_forecast = []
for item in result_final.Item_ID.unique():
    t = result_final[result_final["Item_ID"] == item].copy()
    t.sort_values(by =["MAPE"], ascending=True, inplace=True)
    best_forecast.append(t.index[0])
    
result_best = result_final[result_final.index.isin(best_forecast)].reset_index().drop("index", axis=1)

#--------------------- Forecast out 

df_sarima_division.drop("Division_ID", axis=1, inplace=True)
df_holt_division.drop("Division_ID", axis=1, inplace=True)
df_prophet_division.drop("Division_ID", axis=1, inplace=True)
df_sarima_cluster.drop("Cluster_ID", axis=1, inplace=True)
df_holt_cluster.drop("Cluster_ID", axis=1, inplace=True)
df_prophet_cluster.drop("Cluster_ID", axis=1, inplace=True)


df_sarima_division["Agg_Level"] = "Division"
df_sarima_cluster["Agg_Level"] = "Cluster"
df_holt_division["Agg_Level"] = "Division"
df_holt_cluster["Agg_Level"] = "Cluster"
df_prophet_division["Agg_Level"] = "Division"
df_prophet_cluster["Agg_Level"] = "Cluster"

df_sarima = pd.concat((df_sarima_division, df_sarima_cluster), axis=0).rename({"Item_forecast_Sarima":"Pred"}, axis=1)
df_holt = pd.concat((df_holt_division, df_holt_cluster), axis=0).rename({"Item_forecast_Holt":"Pred"}, axis=1)
df_prophet = pd.concat((df_prophet_division, df_prophet_cluster), axis=0).rename({"Item_Forecast_Prophet":"Pred"}, axis=1)

df_sarima["Forecast"] = "SARIMA"
df_holt["Forecast"] = "Holt Winter"
df_prophet["Forecast"] = "Prophet"

df_out = pd.concat((df_sarima, df_holt), axis=0)
df_out = pd.concat((df_out, df_prophet), axis=0)
df_out = pd.concat((df_out, df_rf), axis=0)
df_out= pd.merge(df_out, result_best, on=["Item_ID", "Agg_Level", "Forecast"], how="left").dropna()

train_len = datetime.datetime(2019,11,1)
data_final = df_items.copy().rename(columns = {"Ship_Qty":"Train"})
data_final["Test"] = data_final.apply(lambda x: x.Train if x.Ship_Date >= train_len else np.nan, axis=1)
data_final["Train"] = data_final.apply(lambda x: x.Train if x.Ship_Date < train_len else np.nan, axis=1)

t = df_out[["Item_ID", "Ship_Date", "Pred"]]
data_final = pd.merge(data_final, t, on=["Item_ID", "Ship_Date"], how="left")
#for item in result_best.Item_ID().values:
    
data_final.to_csv("Data/Output.csv", index=False)
result_best.to_csv("Data/Results.csv", index=False)
result_final.to_csv("Data/ResultsAll.csv", index=False)    
