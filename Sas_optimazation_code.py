#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Importing the required packages
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab  import rcParams
rcParams['figure.figsize']= 10,6


# In[2]:



## reading the data
dt = pd.read_csv("C:/Users/scvch/Downloads/RawData.csv")
cooling = dt[dt['metername'] == 'Bldg. T (Cooling)']
Main = dt[dt['metername'] == 'Bldg. T (Main)']


# In[3]:


from statsmodels.tsa.stattools import adfuller

#### Checking the stationarity in the data that is if the mean, variance and autocorrelation is same across the data
def test_stationarity(timeseries):
    
    #Determing rolling statistics
#     rolmean = pd.rolling_mean(timeseries, window=12)
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(window=24).std()
#     rolstd = timeseries.rolling(12).mean()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='gold', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[7]:


## Checking the stationarity and plotting the data with rolling mean and rolling standard deveiation
test_stationarity(cooling.gallons)


# In[8]:


## Checking the stationarity and plotting the data with rolling mean and rolling standard deveiation
test_stationarity(Main.gallons)


# In[9]:


first_diff = cooling['gallons']-cooling['gallons'].shift(1)
test_stationarity(first_diff[1:])


# In[10]:


cooling.plot()
pyplot.show()


# In[11]:


Main.plot()
pyplot.show()


# In[ ]:





# In[12]:


from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
fft = fft(cooling['gallons'].values)
plt.plot(np.abs(fft))


# In[13]:


plt.plot(np.abs(fft[10:40]))


# In[ ]:





# In[15]:


from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt

fft_main = fft(Main['gallons'].values)
plt.plot(np.abs(fft_main))


# In[16]:


plt.plot(np.abs(fft_main[10:40]))


# In[17]:


plt.plot(np.abs(fft_main[10:50]))


# In[ ]:





# In[ ]:





# ## Cooling Water usage analysis

# In[19]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(cooling['gallons'],lags=30)
plot_pacf(first_diff.iloc[1:],lags=30)


# In[20]:


cooling_train = cooling['gallons'].loc[0:70]
cooling_test = cooling['gallons'].loc[71:]


# ## ARMA Model

# In[22]:


from statsmodels.tsa.arima_model import ARMA
model=ARMA(cooling_train,order=(2,0))
model_fit= model.fit()
model_fit.summary()
forecast = model_fit.predict(start = 71, end = 93, dynamic =True)
cooling['gallons'].plot()
forecast.plot()

pred = model_fit.predict(start = 71, end = 92, dynamic =True)
from sklearn.metrics import mean_squared_error
from math import sqrt
cooling['gallons'].mean()
rmse=sqrt(mean_squared_error(pred,cooling_test))
print(rmse)


# ## Auto regressive model

# In[24]:


model=ARMA(cooling_train,order=(2,0))
model_fit_AR= model.fit()
model_fit_AR.summary()
forecast = model_fit_AR.predict(start = 70, end = 92, dynamic =True)
cooling['gallons'].plot()
forecast.plot()

pred = model_fit_AR.predict(start = 71, end = 92, dynamic =True)
from sklearn.metrics import mean_squared_error
from math import sqrt
cooling['gallons'].mean()
rmse=sqrt(mean_squared_error(pred,cooling_test))
print(rmse)


# In[ ]:





# ## Moving average

# In[26]:


model=ARMA(cooling_train,order=(0,2))
model_fit_MA= model.fit()
model_fit_MA.summary()
forecast = model_fit_MA.predict(start = 70, end = 92, dynamic =True)
cooling['gallons'].plot()
forecast.plot()

pred = model_fit_MA.predict(start = 71, end = 92, dynamic =True)
from sklearn.metrics import mean_squared_error
from math import sqrt
cooling['gallons'].mean()
rmse=sqrt(mean_squared_error(pred,cooling_test))
print(rmse)


# ## ARMA Model

# In[27]:


model=ARMA(cooling_train,order=(2,2))
model_fit= model.fit()
model_fit.summary()
forecast = model_fit.predict(start = 70, end = 92, dynamic =True)
cooling['gallons'].plot()
forecast.plot()

pred = model_fit.predict(start = 71, end = 92, dynamic =True)
from sklearn.metrics import mean_squared_error
from math import sqrt
cooling['gallons'].mean()
rmse=sqrt(mean_squared_error(pred,cooling_test))
print(rmse)


# ## Arima Model

# In[28]:



from statsmodels.tsa.arima_model import ARIMA
model1 = ARIMA(cooling_train,order=(2,0,2))
model_fit1=model1.fit()
forecast = model_fit1.predict(start = 70, end = 92, dynamic =True)
cooling['gallons'].plot()
forecast.plot()

pred = model_fit1.predict(start = 71, end = 92, dynamic =True)
from sklearn.metrics import mean_squared_error
from math import sqrt
cooling['gallons'].mean()
rmse=sqrt(mean_squared_error(pred,cooling_test))
print(rmse)


# ## Sarimax Model :

# In[32]:


import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(cooling_train,order=(2,0,2),seasonal_order=(2,0,2,4))
model_fit3=model.fit()

forecast = model_fit3.predict(start = 71, end = 92, dynamic =True)

cooling['gallons'].plot()
forecast.plot()

pred = model_fit3.predict(start = 71, end = 92, dynamic =True)
from sklearn.metrics import mean_squared_error
from math import sqrt
cooling['gallons'].mean()
rmse=sqrt(mean_squared_error(pred,cooling_test))
print(rmse)


# In[ ]:





# ## Final Model selection

# In[33]:


import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(cooling['gallons'],order=(2,0,2),seasonal_order=(2,0,2,4))
model_fit3=model.fit()

forecast = model_fit3.predict(start = 71, end = 92, dynamic =True)

cooling['gallons'].plot()
forecast.plot()


# In[34]:


forecast2 = model_fit3.predict(start = 93, end = 96, dynamic =True)
cooling['gallons'][-20:].plot()
forecast2


# In[ ]:





# In[ ]:





# ## Employee water usage

# In[35]:


first_diff = Main['gallons']-Main['gallons'].shift(1)
first_diff.plot()

first_diff= first_diff.loc[94:]
#second_diff = first_diff - first_diff.shift(1)
#second_diff.plot()

#third_diff = second_diff - second_diff.shift(1)
#third_diff.plot()


# In[36]:


test_stationarity(first_diff)


# In[37]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(first_diff.iloc[1:],lags=30)
plot_pacf(first_diff.iloc[1:],lags=30)


# In[40]:


# P= 2 and q =2
Main_tank = Main['gallons'].reset_index()


Main_train = Main_tank['gallons'].loc[0:70]
Main_test = Main_tank['gallons'].loc[71:]


# ## Auto correlation

# In[42]:


from statsmodels.tsa.arima_model import ARMA
model=ARMA(Main_train,order=(0,2))
model_fit= model.fit()
model_fit.summary()
forecast = model_fit.predict(start = 71, end = 92, dynamic =True)
Main_tank['gallons'].plot()
forecast.plot()


pred = model_fit.predict(start = 71, end = 92, dynamic =True)
from sklearn.metrics import mean_squared_error
from math import sqrt
Main_tank['gallons'].mean()
rmse=sqrt(mean_squared_error(pred,Main_test))
print(rmse)


# ## Moving Averages
# 

# In[44]:


from statsmodels.tsa.arima_model import ARMA
model=ARMA(Main_train,order=(0,2))
model_fit= model.fit()
model_fit.summary()
forecast = model_fit.predict(start = 71, end = 92, dynamic =True)
Main_tank['gallons'].plot()
forecast.plot()


pred = model_fit.predict(start = 71, end = 92, dynamic =True)
from sklearn.metrics import mean_squared_error
from math import sqrt
Main_tank['gallons'].mean()
rmse=sqrt(mean_squared_error(pred,Main_test))
print(rmse)


# In[ ]:





# ## ARMA

# In[45]:


from statsmodels.tsa.arima_model import ARIMA
model1 = ARIMA(Main_train,order=(2,0,2))
model_fit_ARMA =model1.fit()
forecast = model_fit_ARMA.predict(start = 71, end = 92, dynamic =True)
Main_tank['gallons'].plot()
forecast.plot()

pred = model_fit_ARMA.predict(start = 71, end = 92, dynamic =True)
from sklearn.metrics import mean_squared_error
from math import sqrt
Main_tank['gallons'].mean()
rmse=sqrt(mean_squared_error(pred,Main_test))
print(rmse)


# ## ARIMA

# In[47]:


from statsmodels.tsa.arima_model import ARIMA
model1 = ARIMA(Main_train,order=(2,1,2))
model_fit_ARIMA =model1.fit()
forecast = model_fit_ARIMA.predict(start = 71, end = 92, dynamic =True)
Main_tank['gallons'].plot()
forecast.plot()

pred = model_fit_ARIMA.predict(start = 71, end = 92, dynamic =True)
from sklearn.metrics import mean_squared_error
from math import sqrt
Main_tank['gallons'].mean()
rmse=sqrt(mean_squared_error(pred,Main_test))
print(rmse)


# In[ ]:





# ## SARIMAX

# In[48]:


import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(Main_train,order=(2,1,2),seasonal_order=(2,1,2,4))
model_fit_Sarimax=model.fit()
forecast = model_fit_Sarimax.predict(start = 70, end = 90, dynamic =True)
Main_tank['gallons'].plot()
forecast.plot()


pred = model_fit_Sarimax.predict(start = 71, end = 92, dynamic =True)
from sklearn.metrics import mean_squared_error
from math import sqrt
Main_tank['gallons'].mean()
rmse=sqrt(mean_squared_error(pred,Main_test))
print(rmse)


# In[ ]:





# In[ ]:





# In[49]:


import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(Main_tank['gallons'],order=(2,1,2),seasonal_order=(2,1,2,4))
model_fit_Sarimax=model.fit()
forecast = model_fit_Sarimax.predict(start = 70, end = 90, dynamic =True)
Main_tank['gallons'].plot()
forecast.plot()


# In[50]:


forecast2 = model_fit_Sarimax.predict(start = 93, end = 96, dynamic =True)
forecast2


# In[51]:


model_fit_Sarimax.summary()

