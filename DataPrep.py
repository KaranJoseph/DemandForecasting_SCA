# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:36:45 2022

@author: Yikes
"""

import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

#%matplotlib inline

df = pd.read_excel("Data/Data_Campbell.xlsx", sheet_name= "DataActual")

grouped_location = df[["Location_ID", "Ship_Date", "Ship_Qty"]].\
    groupby(["Location_ID", "Ship_Date"]).sum()
    
grouped_division = df[["Division", "Ship_Date", "Ship_Qty"]].\
    groupby(["Division", "Ship_Date"]).sum()

t1 = df.groupby(["Item_ID", "Division"]).agg(Total_Qty_Item = ("Ship_Qty", "sum")).reset_index()
t2 = df.groupby(["Division"]).agg(Total_Qty_Division = ("Ship_Qty", "sum")).reset_index()
t = pd.merge(t1,t2,how='left',on="Division")
t["scaling_factor"] = t["Total_Qty_Item"] / t["Total_Qty_Division"] #Scaling factor as percentage of actual cummulative demand

t = t.sort_values(["Total_Qty_Item", "Division"], ascending=False)\
    .reset_index().drop("index", axis=1)
top_5 = {}
divisions = list(t["Division"].unique())

for div in divisions:
    sku = list(t[t["Division"] == div]["Item_ID"][:5])
    top_5[div] = sku
    
grouped_location.to_csv("Data/Location.csv")
grouped_division.to_csv("Data/Division.csv")
    
# 1. use grouped_location for forecasting based on location aggregation
# 2. use grouped_division for forecasting based on division aggregation
# 3. top_5 - dictionary of top 5 items by demand contribution in their respective divisions
# 4. The idea is to forecast based on 1 and 2 then dissagregate into the respective item level forecast values based on scaling_factor
# 5. Calculate RMSE, MAPE, cov.. whatever for these 5 items for each division to compare our model accuracy


# 6. Perform time-series clustering and use the new groups to do forecasting
# 7. Compare against previous methods to check forecast accuracy
# 8. Do a min-max inventory system to calculate the cost-benefit for adopting our method