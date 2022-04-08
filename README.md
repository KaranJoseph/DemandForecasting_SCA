# DemandForecasting and InventoryOptimization- Supply Chain Analytics

The goal of the project is to develop a demand forecasting tool to find the best levels of aggregation and the best forecasting technique that can be used to predict weekly demands for a leading Food Processing Company based in North America. The company has several warehouses across USA and Canada, which they use to source retailers and distribution centers across North America. As a complementary service, we have also developed an inventory optimization tool using Integer Linear Programming. Our aim is to improve their production plan for the coming horizons by giving them better forecasts and streamline their inventory levels. 

The company have given us a partial database, with information concerning the actual demand history (univariate time-series dataset) for all the products manufactured in the year 2019. The data at hand is composed of 2407 SKUs, and each of these items fall into one of 4 unique production division categories (CA-FS, CA-RTL, US-FS, and US-RTL). To decrease the variability in the dataset and computational complexity, the company suggested developing forecast models at aggregated levels and then later disaggregate them to get individual product demands for each SKU’s.  

Division level category assignment is being made by the company based on business intuition and product sense. But since SKU’s having different trends and seasonality can get aggregated together under the same division, we believe this will decrease the overall influence of the time series components and thereby reducing its forecasting power. From our research we learned that time-series clustering can be an alternative solution to this problem, because this enabled us to group together items having similar demand patterns. Therefore, we are performing demand forecasting on two different levels of aggregation – **1) Division level and 2) Cluster level** – for building our predictive models.  

In order to analyze the forecast results and develop our prescriptive inventory solution, we have used only the top 5 items by quantity from each divisions (20 items in total). We based this idea off the popular Pareto principle, and we have assumed that this will give us a near accurate representation of the overall data. The forecast models were then developed using 80% of data as the train set and the remaining 20% as the test set. The overall performance of models were compared using the test-set and we have used this half of the data for the inventory model. The inventory optimization model was developed using an Integer Linear Programming algorithm for the last 8 weeks of 2019 (test set).   

# Folder Descriptions:

1) Adhoc (Not using in Report) -> Adhoc scripts which we used to check for outliers, initial forecasting, and time-series plots.
*(Please note that the codes in this folder may not run in your system, as it is not structured to be in an executable format - these are some additional codes which is not required for the pipeline)*

2) ***Codes (Main codes)***
	- a) DataPrep.py - Run this script to prepare the division level and cluster level aggregations
			     - Basic data cleaning, sanity checks, and time-series clustering
	- b) RandomForest_TimeSeries.py - Run this script to get the outputs from RandomForest (results and forecasts)
	- c) TimeSeriesForecasting - Run this script to get the output from all other Forecast techniques used (results and forecasts)
	- d) InventoryManagement - Use this file to get the InventoryManagementOutput
					 - Please copy the **Inventory.csv** downloaded into the **Data Folder** to reflect the changes made
4) ***Data - Data Inputs and Outputs***
5) ***Images - Images used in the report***
6) OutImages - Adhoc checks


**Please refresh the dashboard after running the codes in "Codes" folder to see the output of your changes made**


