# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:44:07 2022

@author: Yikes
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_cluster = pd.read_csv("Data/Cluster.csv")
df_division = pd.read_csv("Data/Division.csv")
df_item = pd.read_csv("Data/Item.csv")
df_scaling = pd.read_csv("Data/Scaling.csv")


def outlier_detection(df, column):
    """
    Parameters
    ----------
    df : DataFrame
        Input df_item, df_cluster, and df_division.
    column : String
        Use "Item_ID", "Cluster_ID", and "Division" depending on input df.

    Returns
    -------
    perc_outliers : Dict
        Returns number of outliers for each unique elements of the df.
    """
    perc_outliers={}
    out_index={}
    df = df.set_index("Ship_Date")
    for i,j in enumerate(df[column].unique()):
        x = df[df[column] == j]["Ship_Qty"]
        IQR = np.percentile(x,75) - np.percentile(x,25)
        Q3 = np.percentile(x,75)
        Q1 = np.percentile(x,25)
        Upper_bound = (x>Q3+1.5*IQR).sum()
        Lower_bound = (x<Q1-1.5*IQR).sum()
        perc_outliers[j] = (Upper_bound + Lower_bound)
        out_index[j] = list(x[(x>Q3+1.5*IQR) | (x<Q1-1.5*IQR)].index)
            
        
    fig1 = plt.figure(len(df[column].unique()), figsize=(50,60))
    for i,j in enumerate(df[column].unique()):
        plt.subplot(10,2,i+1)
        df[df[column] == j].boxplot(column=['Ship_Qty'])
        plt.title(j)
    fig1.savefig(f'OutImages/{column}/Boxplot{column}.png')
    plt.clf()
        
    fig2 = plt.figure(len(df[column].unique()), figsize=(50,60))
    for i,j in enumerate(df[column].unique()):
        plt.subplot(10,2,i+1)
        temp = df[df[column] == j]
        plt.plot(temp.index, temp["Ship_Qty"])
        temp = temp[temp.index.isin(out_index[j])]
        plt.scatter(temp.index, temp["Ship_Qty"], color="red")
        plt.title(j)
        plt.xticks([])
    fig2.savefig(f'OutImages/{column}/Lineplot{column}.png')
    plt.clf()
        
    fig3 = plt.figure(len(df[column].unique()), figsize=(50,60))
    for i,j in enumerate(df[column].unique()):
        plt.subplot(10,2,i+1)
        sns.distplot(df[df[column] == j]["Ship_Qty"])
        plt.title(j)
    fig3.savefig(f'OutImages/{column}/Distplot{column}.png')
    plt.clf()
        
    return perc_outliers

print(outlier_detection(df_division, "Division"))
print(outlier_detection(df_cluster, "Cluster_ID"))
print(outlier_detection(df_item, "Item_ID"))


def outlier_correction(df):
    """
    Parameters
    ----------
    df : Dataframe
         Use item level data as input.

    Returns
    -------
    df : Dataframe
         Fixes outliers for each item in the dataframe.
    """
    items = df["Item_ID"].unique()
    for i in items:
        x = df[df["Item_ID"] == i]["Ship_Qty"]
        IQR = np.percentile(x,75) - np.percentile(x,25)
        Q3 = np.percentile(x,75)
        Q1 = np.percentile(x,25)
        item_outliers = list(x[(x>Q3+1.5*IQR) | (x<Q1-1.5*IQR)].index)
        df.loc[(df.Item_ID==i) & (df.index.isin(item_outliers)), "Ship_Qty"] = np.nan
        df.loc[df.Item_ID==i, "Ship_Qty"] = df.loc[df.Item_ID==i, "Ship_Qty"].interpolate(method="linear")
    return df

df = outlier_correction(df_item)
print(outlier_detection(df, "Item_ID"))
    
