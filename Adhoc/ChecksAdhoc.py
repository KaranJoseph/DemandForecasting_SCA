# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 00:18:58 2022

@author: Yikes
"""

import pandas as pd
df = pd.read_csv("Data/Division.csv", parse_dates=True)
df["Ship_Date"] = pd.to_datetime(df["Ship_Date"])
df = df[df["Division_ID"] == "CA-RTL"]
df.set_index("Ship_Date", inplace=True)
df.sort_index(ascending=True, inplace=True)
s = df["Ship_Qty"]

from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(s, model='additive', period=21)
result.plot()
pyplot.show()


from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm


from fbprophet import Prophet
m = Prophet()
data = pd.DataFrame(s).reset_index().rename(columns={"Ship_Date":"ds", "Ship_Qty":"y"})
m.add_seasonality(name='weekly', period=13, fourier_order=3)
m.fit(data.iloc[:45,:])
forecast = m.predict(data.iloc[45:,:])
m.plot_components(forecast)


i=0
s = [i for i in range(0,26)]
P = [i for i in range(0,3)]
Q = [i for i in range(0,3)]
D = [i for i in range(0,3)]
df1=pd.DataFrame(columns=["AIC","Attributes"])
import statsmodels.api as sm
for d in D:
    for p in P:
        for q in Q:
            for param in s:
                try:
                        mod = sm.tsa.statespace.SARIMAX(df.iloc[:45, 1].values,
                                         order=[1,1,0],
                                         seasonal_order=[p,d,q,param])
                        results = mod.fit(max_iter = 50, method = 'powell')
                        df1.loc[i,"AIC"]=results.aic
                        add_list=[p,d,q,param]
                        df1.loc[i,"Attributes"]=add_list 
                except:
                    continue
                i=i+1

[0,2,1,13]

x = pd.concat([forecast.yhat, s[45:].reset_index().drop("Ship_Date", axis=1)], axis=1)
x.plot()


df1.to_csv("test.csv",index=False)
