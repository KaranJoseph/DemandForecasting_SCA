# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:36:45 2022

@author: Yikes!!
"""
"""
# 1. use grouped_location for forecasting based on location aggregation
# 2. use grouped_division for forecasting based on division aggregation
# 3. top_5 - dictionary of top 5 items by demand contribution in their respective divisions
# 4. The idea is to forecast based on 1 and 2 then dissagregate into the respective item level forecast values based on scaling_factor
# 5. Calculate RMSE, MAPE, cov.. whatever for these 5 items for each division to compare our model accuracy
# 6. Perform time-series clustering and use the new groups to do forecasting
# 7. Compare against previous methods to check forecast accuracy
# 8. Do a min-max inventory system to calculate the cost-benefit for adopting our method
"""
import pandas as pd
import numpy as np
import datetime
from tslearn.metrics import dtw
import matplotlib.pyplot as plt
import os 
os.chdir('..')
#import seaborn as sns

#%matplotlib inline
#-------------- Data Cleaning------------------
df = pd.read_excel("Data/Data_Campbell.xlsx", sheet_name= "DataActual")
df["Ship_Date"] = pd.to_datetime(df["Ship_Date"])
df = df[df["Ship_Date"] >= datetime.datetime(2019,1,1)] #Only 2019

df["Ship_Qty"].describe() # No negatives
df.isna().sum() # No nulls
    
data_ts = df.groupby(["Item_ID", "Ship_Date", "Division"])["Ship_Qty"].sum().reset_index()

# drop items without 52 weeks of demand data
# For the sake of the simplicity I'm ignoring all items without 52 weeks of demand
item_remove = []
temp = data_ts.groupby("Item_ID").agg(counts = ("Ship_Date", "count"))
item_remove = list(temp[temp["counts"] != 52].index)
df = df[~df["Item_ID"].isin(item_remove)]
data_ts = data_ts[~data_ts["Item_ID"].isin(item_remove)]

#----------------------END----------------------

#grouped_location = df[["Location_ID", "Ship_Date", "Ship_Qty"]].\
    #groupby(["Location_ID", "Ship_Date"]).sum()
    
grouped_division = data_ts[["Division", "Ship_Date", "Ship_Qty"]].\
    groupby(["Division", "Ship_Date"]).sum()

t1 = data_ts.groupby(["Item_ID", "Division"]).agg(Total_Qty_Item = ("Ship_Qty", "sum")).reset_index()
t2 = data_ts.groupby(["Division"]).agg(Total_Qty_Division = ("Ship_Qty", "sum")).reset_index()
t = pd.merge(t1,t2,how='left',on="Division")
t["Scaling_Factor_Division"] = t["Total_Qty_Item"] / t["Total_Qty_Division"] #Scaling factor as percentage of actual cummulative demand

t = t.sort_values(["Total_Qty_Item", "Division"], ascending=False)\
    .reset_index().drop("index", axis=1)
    
top_5 = {}
divisions = list(t["Division"].unique())

for div in divisions:
    """
    Get top 5 items for each Divisions by qty contribution
    """
    sku = list(t[t["Division"] == div]["Item_ID"][:5])
    top_5[div] = sku
    
#grouped_location.to_csv("Data/Location.csv")
grouped_division.reset_index().rename(columns={"Division":"Division_ID"}).to_csv("Data/Division.csv", index=False)


##### Time Series Clustering ####

val = []
#Normalize the time-series data for each Item_ID
for item in data_ts["Item_ID"].unique():
    """
    Normalize Ship_Qty for each Item_ID and store it in Qty_Normalized
    Min-Max Scaling
    """
    x = np.array(data_ts[data_ts["Item_ID"] == item]["Ship_Qty"])
    x = (x-min(x))/(max(x)-min(x))
    val.extend(x)
data_ts["Qty_Normalized"] = val

from dtaidistance import dtw, clustering
from dtaidistance import dtw_visualisation as dtwvis
from scipy.cluster.hierarchy import single, dendrogram, fcluster, complete, ward
from tslearn.clustering import TimeSeriesKMeans,silhouette_score

s1 = np.array(data_ts[data_ts["Item_ID"] == 1269265]["Qty_Normalized"])[:12]
s2 = np.array(data_ts[data_ts["Item_ID"] == 1270709]["Qty_Normalized"])[:12]
path = dtw.warping_path(s1, s2)
dtwvis.plot_warping(s1, s2, path, filename="Images/dtw_example1.png")

s3 = np.sin(np.arange(0, 20, .5))
s4 = np.sin(np.arange(0, 20, .5) - 2)
path = dtw.warping_path(s3, s4)
dtwvis.plot_warping(s3, s4, path, filename="Images/dtw_example2.png")

"""
Clustering is used to find groups of similar instances (e.g. time series, sequences). Such a clustering can be used to:
- There is information in the series order
- Different series may have similar patterns that are not aligned in time
- Series may vary in length
"""
ts_pivot = data_ts.pivot(index="Item_ID", columns="Ship_Date", values="Qty_Normalized")
dtw_ds = dtw.distance_matrix_fast(ts_pivot.to_numpy())

"""
Heirarchical clustering - dtw single linkage 
"""

model1=ward(dtw_ds)
fig = plt.figure(figsize=(16, 8))
dn1 = dendrogram(model1)
plt.title("Dendrogram for ward-linkage with correlation distance")
plt.savefig("Images/agglomerativeClusteringWard.png")
cluster_heirarchical = list(fcluster(model1, 4, criterion='maxclust'))
print(silhouette_score(ts_pivot, cluster_heirarchical, "dtw"))

plt.clf()

model2 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
cluster_idx = model2.fit(ts_pivot.to_numpy())
model3 = clustering.HierarchicalTree(model2)
cluster_idx = model3.fit(ts_pivot.to_numpy())

fig = plt.figure(figsize=(16, 8))
dn2 = dendrogram(model3.linkage)
plt.title("Dendrogram for default-linkage with correlation distance")
plt.savefig("Images/agglomerativeClusteringDefault.png")
cluster_idx = list(fcluster(model3.linkage, 1.5, criterion="distance"))
print(silhouette_score(ts_pivot, cluster_idx, "dtw"))
plt.clf()
"""
KMeans clustering based on dtw 
"""

#Silhouette score
ssd1=[]
ssd2=[]
for i in range(2,9):
    kmeans=TimeSeriesKMeans(n_clusters=i,\
                            metric="dtw",\
                            max_iter=5,\
                            max_iter_barycenter=5,\
                            #metric_params={"gamma": .5},\
                            random_state=0)
    kmeans.fit(ts_pivot)
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(ts_pivot, cluster_labels, metric="dtw")
    print("For n_clusters={0}, the silhouette score is {1}".format(i, silhouette_avg))
    ssd1.append([i,silhouette_avg])
    ssd2.append((i,kmeans.inertia_))
    
plt.plot(pd.DataFrame(ssd1)[0], pd.DataFrame(ssd1)[1])
plt.savefig("Images/kmeansSilhouetteScore.png")
plt.clf()

# Elbow Curve
plt.plot(pd.DataFrame(ssd2)[0], pd.DataFrame(ssd2)[1])
plt.savefig("Images/kmeansElbowCurve.png")
plt.clf()

km_sdtw = TimeSeriesKMeans(n_clusters=4,\
                           metric="dtw",\
                           max_iter=10,\
                           max_iter_barycenter=10,\
                           #metric_params={"gamma": .5},\
                           random_state=0).fit(ts_pivot)    
cluster_kmeans = list(km_sdtw.predict(ts_pivot))


def plot_clusters(df, cluster_labels, title, row_mx=2, column_mx=2):
    """
    Plot the time-series cluster groups based on the method
    Parameters: 
        df : (dataframe) Input time-series dataframe used for clustering.
        cluster_labels : (list)Ouput of cluster model.
        title : (string) Name of clustering model.
        row_mx : (int, optional) Number of rows to show in subplot. The default is 2.
        column_mx : (int, optional) Number of columns in subplot. The default is 2.

    Returns:
        Outputs cluster plots as png and saves them in the Images folder
    """
    fig, axs = plt.subplots(row_mx, column_mx, figsize=(25,25))
    fig.suptitle('Clusters')
    loc = []
    row = 0
    column = 0
    for i in set(cluster_labels):
        loc = [x for x,y in enumerate(cluster_labels) if y==i]
        for j in range(10):
            try:
                axs[row, column].plot(df.iloc[loc[j],:].values, c="gray", alpha=0.4)
            except:
                axs[row, column].plot(df.iloc[loc,:].values, c="gray", alpha=0.4)
        if j>1:
            axs[row, column].plot(np.average(np.vstack(df.iloc[loc,:].values), axis=0),c="red")        
        column+=1
        if column>=column_mx:
            row+=1
            column=0
    plt.savefig(f'Images/{title}.png')
    
plot_clusters(ts_pivot, cluster_kmeans, "kmeans")
plot_clusters(ts_pivot, cluster_heirarchical, "agglomerativeWard")
plot_clusters(ts_pivot, cluster_idx, "agglomerativeDefault", 3, 2)

# Based on the cluser plots and silhouette score, kmeans is giving the better clustering results. 
# So our cluster based  forecasting will be based on this one group

cluster_result = pd.concat((pd.Series(ts_pivot.index), pd.Series(cluster_kmeans, name="Cluster_ID")), axis=1)
cluster_result = pd.merge(cluster_result, data_ts.groupby("Item_ID").agg(Total_Qty_Item = ("Ship_Qty", "sum")).reset_index(),\
                          on="Item_ID")
cluster_result = pd.merge(cluster_result,cluster_result.groupby("Cluster_ID")\
                         .agg(Total_Qty_Cluster = ("Total_Qty_Item", "sum")).reset_index(), on="Cluster_ID")
    
cluster_result["Cluster_Scaling_Factor"] = cluster_result["Total_Qty_Item"] / cluster_result["Total_Qty_Cluster"]

df_scale = pd.merge(t, cluster_result, on="Item_ID")[["Item_ID", "Division", "Cluster_ID",\
                                                      "Scaling_Factor_Division", "Cluster_Scaling_Factor"]]

df_scale.rename(columns= {"Division":"Division_ID", "Scaling_Factor_Division":"Division_Scaling_Factor"}, inplace=True)
items_include = []
for i in list(top_5.values()):
    items_include.extend(i) 
df_scale = df_scale[df_scale["Item_ID"].isin(items_include)]
df_scale.to_csv("Data/Scaling.csv", index=False)

data_ts[data_ts["Item_ID"].isin(items_include)].groupby(["Item_ID","Ship_Date"])\
    .agg(Ship_Qty = ("Ship_Qty", "sum")).reset_index()\
    .to_csv("Data/Item.csv", index=False)

pd.merge(data_ts, cluster_result[["Item_ID", "Cluster_ID"]], on="Item_ID").groupby(["Cluster_ID", "Ship_Date"])\
    .agg(Ship_Qty = ("Ship_Qty", "sum")).reset_index()\
    .to_csv("Data/Cluster.csv", index=False)