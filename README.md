# DemandForecasting and InventoryOptimization- Supply Chain Analytics

The goal of the project is to perform demand forecasting at different aggregation levels (product divisions and locations) on the demand data for 2018-2019 of a Food processing Company that sources its products to various distribution centers in the US and Canada. With the improved forecasting models, the company will be able to prepare a better production plan for the next horizon and reduce inventory costs.

The data we are using has the actual demand history of all the products sold by the company over the years 2018-2019 by SKU and location. We also have an Item Master that describes which item belongs to which division/category of products. To reduce variability in the dataset, we are planning to aggregate the demand by division and location before doing our predictive modeling. We are mainly planning to use multiple forecasting models and then compare the accuracy among them to identify the best models by product divisions and location. 

The various evident steps that we have to perform based on the dataset, is to first clean and aggregate the dataset to the format that we are planning to use in our models. Then we are going to perform descriptive analytics and data visualizations. The final step is to run the data through the different forecasting models and compare the results based on forecasting accuracy...


# Folder Descriptions:

1) Adhoc (Not using in Report) -> Adhoc scripts which we used to check for outliers, initial forecasting, and time-series plots

2) ***Codes (Main codes)***
	- a) DataPrep.py - Run this script to prepare the division level and cluster level aggregations
			     - Basic data cleaning, sanity checks, and time-series clustering
	- b) GridSearch - For finding the attributes of SARIMA ("Donot run the code" - computationally expensive)
	- c) RanfomForest_TimeSeries.py - Run this script to get the outputs from RandomForest (results and forecasts)
	- d) TimeSeriesForecasting - Run this script to get the output from all other Forecast techniques used (results and forecasts)
	- e) InventoryManagement - Use this file to get the InventoryManagementOutput
					 - Please copy the **Inventory.csv** downloaded into the **Data Folder** to reflect the changes made
4) ***Data - Data Inputs and Outputs***
5) ***Images - Images used in the report***
6) OutImages - Adhoc checks


**Please refresh the dashboard after running the codes in "Codes" folder to see the output of your changes made**

