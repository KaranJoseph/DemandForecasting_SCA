# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:36:45 2022

@author: Yikes
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
#import seaborn as sns

#%matplotlib inline
#-------------- Data Cleaning------------------
df = pd.read_excel("Data/Data_Campbell.xlsx", sheet_name= "DataActual")
df["Ship_Date"] = pd.to_datetime(df["Ship_Date"])
df = df[df["Ship_Date"] >= datetime.datetime(2019,1,1)] #Only 2019

data_ts = df.groupby(["Item_ID", "Ship_Date"])["Ship_Qty"].sum().reset_index()

# drop items without 52 weeks of demand data
# For the sake of the simplicity I'm ignoring all items without 52 weeks of demand
item_remove = []
temp = data_ts.groupby("Item_ID").agg(counts = ("Ship_Date", "count"))
item_remove = list(temp[temp["counts"] != 52].index)
df = df[~df["Item_ID"].isin(item_remove)]
data_ts = data_ts[~data_ts["Item_ID"].isin(item_remove)]

#----------------------END----------------------

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
    """
    Get top 5 items for each Divisions by qty contribution
    """
    sku = list(t[t["Division"] == div]["Item_ID"][:5])
    top_5[div] = sku
    
grouped_location.to_csv("Data/Location.csv")
grouped_division.to_csv("Data/Division.csv")


##### Time Series Clustering ####

val = []
#Normalize the time-series data for each Item_ID
for item in data_ts["Item_ID"].unique():
    """
    Normalize Ship_Qty for each Item_ID and store it in Qty_Normalized
    Min-Max Scaling
    """
    x = np.array(data_ts[data_ts["Item_ID"] == item]["Ship_Qty"])
    x = (x-max(x))/(max(x)-min(x))
    val.extend(x)
data_ts["Qty_Normalized"] = val

from dtaidistance import dtw, clustering
from dtaidistance import dtw_visualisation as dtwvis
from scipy.cluster.hierarchy import single, dendrogram, fcluster, complete
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score

s1 = np.array(data_ts[data_ts["Item_ID"] == 1269265]["Qty_Normalized"])[:12]
s2 = np.array(data_ts[data_ts["Item_ID"] == 1270709]["Qty_Normalized"])[:12]
path = dtw.warping_path(s1, s2)
dtwvis.plot_warping(s1, s2, path, filename="dtw_example1.png")

s3 = np.sin(np.arange(0, 20, .5))
s4 = np.sin(np.arange(0, 20, .5) - 2)
path = dtw.warping_path(s3, s4)
dtwvis.plot_warping(s3, s4, path, filename="dtw_example2.png")

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

model1=complete(dtw_ds)
fig = plt.figure(figsize=(16, 8))
dn1 = dendrogram(model1)
plt.title("Dendrogram for single-linkage with correlation distance")
plt.show()
cluster_heirarchical = list(fcluster(model1, 3, criterion='maxclust'))
silhouette_score(ts_pivot, cluster_heirarchical)


model2 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
cluster_idx = model2.fit(ts_pivot.to_numpy())
model3 = clustering.HierarchicalTree(model2)
cluster_idx = model3.fit(ts_pivot.to_numpy())

dn2 = dendrogram(model3.linkage)
plt.show()

cluster_idx = list(fcluster(model3.linkage, 1.5, criterion="distance"))

"""
KMeans clustering based on dtw 
"""

#Silhouette score
ssd=[]
for i in range(2,9):
    kmeans=TimeSeriesKMeans(n_clusters=i, metric="softdtw",\
                            max_iter=5,max_iter_barycenter=5,\
                            metric_params={"gamma": .5}, random_state=0)
    kmeans.fit(ts_pivot)
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(ts_pivot, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(i, silhouette_avg))
    ssd.append([i,silhouette_avg])
    
plt.plot(pd.DataFrame(ssd)[0], pd.DataFrame(ssd)[1])
plt.show()

#Elbow curve
ssd = []
for num_clusters in list(range(1,8)):
    model_clus = TimeSeriesKMeans(n_clusters = num_clusters, metric="softdtw",\
                                  max_iter=5,max_iter_barycenter=5,\
                                  metric_params={"gamma": .5}, random_state=0)
    model_clus.fit(ts_pivot)
    ssd.append((num_clusters,model_clus.inertia_))

plt.plot(pd.DataFrame(ssd)[0], pd.DataFrame(ssd)[1])
plt.show()


km_sdtw = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=5,\
                           max_iter_barycenter=5,\
                           metric_params={"gamma": .5},\
                           random_state=0).fit(ts_pivot)    
cluster_kmeans = list(km_sdtw.predict(ts_pivot))



