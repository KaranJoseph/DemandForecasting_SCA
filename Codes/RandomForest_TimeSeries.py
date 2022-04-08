# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:47:26 2022

@author: Yikes
"""
import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

import os
os.chdir('..')

df_item = pd.read_csv("Data/Item.csv")
df_item["Ship_Date"] = pd.to_datetime(df_item["Ship_Date"])
df_cluster = pd.read_csv("Data/Cluster.csv")
df_division = pd.read_csv("Data/Division.csv")
scaling = pd.read_csv("Data/Scaling.csv")
    
"""
Very limited due to avaialability of data - if we had atleast 2 yrs of data, we could add seasonal factors such as
week, month etc.. Only thing we could add was lag and that too is limited because bigger lags means we would have to
mask that much data in the final prediction
"""

#-------------------------------------------------------------------------------#

#RMSE AND MAPE
def calculate_rmse(actual,predicted):
  rmse = np.sqrt(mean_squared_error(actual, predicted)).round(2)
  return rmse
def calculate_mape(actual,predicted):
  mape= np.round(np.mean(np.abs(actual-predicted)/actual)*100,2)
  return mape


def random_forest(X_train, y_train, X_test):
    """
    Parameters
    ----------
    X_train : DataFrame
        Predictor variables for model input - train data - in sample
    y_train : DataFrame
        Target variable for model input - train data - in sample
    X_test : DataFrame
        Predictor variables for model input - test data - out of sample 

    Returns
    -------
    y_pred : Series
        Out from the random forest model for input X_test - test set - out of sample prediction

    """
    rfr = RandomForestRegressor(n_estimators=1000, random_state=1)
    rfr.fit(X_train, y_train)
    
    index_start = X_test.index[0]
    y_pred = []
    for i in range(X_test.shape[0]):
        pred = rfr.predict(pd.DataFrame(X_test.iloc[i, :]).T)[0]
        y_pred.append(pred)
        for j in range(1,6):
            try:
                X_test.at[index_start+i+j ,"t-"+str(j)] = pred
            except:
                continue
    return y_pred

def random_forest_prep(df, level):
    """
    Parameters
    ----------
    df : DataFrame
        Receives the quantity data by each item and processes it for input to the random_forest model
    
    level : String
        Level of aggregation being called : Division or Cluster
        
    Returns
    -------
    out : DataFrame 
        Output from random_forest model
    
    """
    
    df["Ship_Date"] = pd.to_datetime(df["Ship_Date"])
    train_len = datetime.datetime(2019,11,1)
    for i in range(5,0,-1):
        df["t-"+str(i)] = df["Ship_Qty"].shift(i)
    df.dropna(inplace=True)
    train = df[df["Ship_Date"]<train_len].copy()
    test = df[df["Ship_Date"]>=train_len].copy()
    X_test = test.iloc[:,3:]
    X_train = train.iloc[:,3:]
    y_train = train.iloc[:,2]
    y_pred = pd.Series(random_forest(X_train, y_train, X_test), name="Qty_Predicted")
    out = pd.concat((test[[level, "Ship_Date", "Ship_Qty"]]\
                     .reset_index().drop("index",axis=1), y_pred), axis=1)
    return out
        

df_division_rf = pd.DataFrame()
for div in df_division.Division_ID.unique():
    t = df_division[df_division["Division_ID"] == div]
    df = random_forest_prep(t, "Division_ID")
    df_division_rf = df_division_rf.append(df)
    
df_cluster_rf = pd.DataFrame()
for cluster in df_cluster.Cluster_ID.unique():
    t = df_cluster[df_cluster["Cluster_ID"] == cluster]
    df = random_forest_prep(t, "Cluster_ID")
    df_cluster_rf = df_cluster_rf.append(df)
    
"""  
d1={}   
for i in df_cluster_rf.Cluster_ID.unique():
    t = df_cluster_rf[df_cluster_rf["Cluster_ID"] == i]
    d1[i] = calculate_mape(t["Ship_Qty"],t["Qty_Predicted"])  

d2={}   
for i in df_division_rf.Division_ID.unique():
    t = df_division_rf[df_division_rf["Division_ID"] == i]
    d2[i] = calculate_mape(t["Ship_Qty"],t["Qty_Predicted"])  
"""   
    
df_item_division = pd.merge(scaling, df_division_rf, on=["Division_ID"], how="left")\
    .rename(columns={"Qty_Predicted":"Qty_Division"})
df_item_division["Qty_Division"] = df_item_division["Qty_Division"] * df_item_division["Division_Scaling_Factor"]
df_item_division = df_item_division[["Item_ID", "Ship_Date", "Qty_Division"]]

df_item_cluster = pd.merge(scaling, df_cluster_rf, on=["Cluster_ID"], how="left")\
    .rename(columns={"Qty_Predicted":"Qty_Cluster"})
df_item_cluster["Qty_Cluster"] = df_item_cluster["Qty_Cluster"] * df_item_cluster["Cluster_Scaling_Factor"]
df_item_combined = pd.merge(df_item_cluster, df_item_division, on=["Item_ID", "Ship_Date"]).drop(columns="Ship_Qty")

df_item_combined = pd.merge(df_item_combined, df_item, on=["Item_ID","Ship_Date"])

def results_rf(df, level):
    """
    Combining the results from cluster based and division based random_forest models
    """
    results = {}
    col = "Qty_"+level
    for item in df.Item_ID.unique():
        t = df[df["Item_ID"] == item]
        results[item] = {}
        results[item]["RMSE"] = calculate_rmse(t["Ship_Qty"], t[col])
        results[item]["MAPE"] = calculate_mape(t["Ship_Qty"], t[col])
    return results

results = {}
results["Division"] =  results_rf(df_item_combined, "Division")
results["Cluster"] =  results_rf(df_item_combined, "Cluster")
    
results_division = pd.DataFrame(results["Division"]).T.reset_index().rename(columns={"index":"Item_ID"})
results_cluster = pd.DataFrame(results["Cluster"]).T.reset_index().rename(columns={"index":"Item_ID"})

results_division["Agg_Level"] = "Division"
results_cluster["Agg_Level"] = "Cluster"
results_df = pd.concat([results_division, results_cluster], axis=0).reset_index().drop(columns='index')
results_df["Forecast"] = "RandomForest"
results_df = results_df[["Item_ID", "Agg_Level", "Forecast", "RMSE", "MAPE"]]
results_df.to_csv("Data/RandomForest_Result.csv", index=False)


t1 = df_item_combined[["Item_ID", "Ship_Date", "Ship_Qty", "Qty_Cluster"]].rename({"Qty_Cluster":"Pred"}, axis=1)
t1["Agg_Level"] = "Cluster"
t1["Forecast"] = "RandomForest"
t2 = df_item_combined[["Item_ID", "Ship_Date", "Ship_Qty", "Qty_Division"]].rename({"Qty_Division":"Pred"}, axis=1)
t2["Agg_Level"] = "Division"
t2["Forecast"] = "RandomForest"

pd.concat((t1,t2), axis=0).reset_index().drop("index", axis=1).to_csv("Data/RandomForest_Out.csv", index=False)


"""
#-----------------------------------------------
train_len = datetime.datetime(2019,11,1)
data_out = df_item.copy().rename(columns = {"Ship_Qty":"Train"})
data_out["Test"] = data_out.apply(lambda x: x.Train if x.Ship_Date >= train_len else np.nan, axis=1)
data_out["Train"] = data_out.apply(lambda x: x.Train if x.Ship_Date < train_len else np.nan, axis=1)


#---------------------------------------------------------PLOTS------------------------------
def plot_forecast(df):
    fig1 = plt.figure(len(df["Item_ID"].unique()), figsize=(50,60))
    for i,j in enumerate(df["Item_ID"].unique()):
        plt.subplot(10,2,i+1)
        df[df["Item_ID"] == j]['Ship_Qty'].plot()
        df[df["Item_ID"] ==j]["Qty_Cluster"].plot()
        df[df["Item_ID"] ==j]["Qty_Division"].plot()
        plt.title(j)
    #fig1.savefig(f'{j}.png')
    plt.clf()



df = pd.merge(df_item, df_item_combined[["Item_ID", "Ship_Date", "Qty_Cluster", "Qty_Division"]],\
             on=['Item_ID','Ship_Date'], how="outer")
    
#plot_forecast(A)

df.set_index("Ship_Date", inplace=True)
fig1 = plt.figure(len(df["Item_ID"].unique()), figsize=(20,50))
train_len = datetime.datetime(2019,11,1)
for i,j in enumerate(df["Item_ID"].unique()):
    plt.subplot(10,2,i+1)
    df[(df["Item_ID"] == j) & (df.index<train_len)]['Ship_Qty'].plot(legend=True, label="Train")
    df[(df["Item_ID"] == j) & (df.index>=train_len)]['Ship_Qty'].plot(legend=True, label="Test")
    df[df["Item_ID"] ==j]["Qty_Cluster"].plot(legend=True, label="Pred Cluster", style="--")
    df[df["Item_ID"] ==j]["Qty_Division"].plot(legend=True, label="Pred Division", style="--")
    plt.title(j)
fig1.savefig('RandomForest1.png')
plt.clf()


df_cluster["Ship_Date"] = pd.to_datetime(df_cluster["Ship_Date"])
df = pd.merge(df_cluster, df_cluster_rf[["Cluster_ID", "Ship_Date", "Qty_Predicted"]],\
              on=["Cluster_ID","Ship_Date"], how="outer")
   
df.set_index("Ship_Date", inplace=True)
fig1 = plt.figure(len(df["Cluster_ID"].unique()), figsize=(15,17))
train_len = datetime.datetime(2019,11,1)
for i,j in enumerate(df["Cluster_ID"].unique()):
    plt.subplot(2,2,i+1)
    df[(df["Cluster_ID"] == j) & (df.index<train_len)]['Ship_Qty'].plot(legend=True, label="Train")
    df[(df["Cluster_ID"] == j) & (df.index>=train_len)]['Ship_Qty'].plot(legend=True, label="Test")
    df[df["Cluster_ID"] ==j]["Qty_Predicted"].plot(legend=True, label="Forecast", style="--")
    plt.title(j)
fig1.savefig('RandomForest2.png')
plt.clf()
    


df_division["Ship_Date"] = pd.to_datetime(df_division["Ship_Date"])
df = pd.merge(df_division, df_division_rf[["Division_ID", "Ship_Date", "Qty_Predicted"]],\
              
              on=["Division_ID","Ship_Date"], how="outer")
   
df.set_index("Ship_Date", inplace=True)
fig1 = plt.figure(len(df["Division_ID"].unique()), figsize=(15,17))
train_len = datetime.datetime(2019,11,1)
for i,j in enumerate(df["Division_ID"].unique()):
    plt.subplot(2,2,i+1)
    df[(df["Division_ID"] == j) & (df.index<train_len)]['Ship_Qty'].plot(legend=True, label="Train")
    df[(df["Division_ID"] == j) & (df.index>=train_len)]['Ship_Qty'].plot(legend=True, label="Test")
    df[df["Division_ID"] ==j]["Qty_Predicted"].plot(legend=True, label="Forecast", style="--")
    plt.title(j)
fig1.savefig('RandomForest3.png')
plt.clf()
    
"""













