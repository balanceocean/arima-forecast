import pandas_datareader as pdr
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import warnings
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
    
warnings.filterwarnings("ignore")

# Download the information of closing prices from Yahoo Finance
start_date = '01-Jan-08'
end_date   = '14-Feb-20'
ticker     = 'V' # Set the ticker (Visa in this example)

data = pdr.get_data_yahoo(ticker, start_date, end_date)['Adj Close']

# Plot the Adjusted Close Price
plt.rcParams["figure.figsize"] = (10, 7)
plt.style.use('seaborn-darkgrid')

data.plot()
plt.title('Visa Stock Prices')
plt.ylabel('Price in Dollars')
plt.legend(['Adj_Close'])
plt.show()

# Density Plot
sns.distplot(data.diff().dropna(), hist=False, kde=True)

# Splitting data into train and test set
split = int(len(data)*0.90)
train_set, test_set = data[:split], data[split:]

plt.title('Visa Stock Price')
plt.xlabel('Date')
plt.ylabel('Price in dollars')
plt.plot(train_set, 'green')
plt.plot(test_set, 'red')
plt.legend(['Training Set','Testing Data'])
plt.show()

# Tuning the AR Term #
aic_p = []
bic_p = []
    
p = range(1,6) # [1,2,3,4,5]
    
# aic/bic score for different values of p
for i in p:
    model = ARIMA(train_set, order=(i,1,0)) # define AR model
    model_fit = model.fit(disp=-1) # fit the model
    aic_temp = model_fit.aic # get aic score
    bic_temp = model_fit.bic # get bic score
    aic_p.append(aic_temp) # append aic score
    bic_p.append(bic_temp) # append bic score
    
# Plot of AIC/BIC score for AR term
plt.plot(range(1,6),aic_p, color='red')
plt.plot(range(1,6),bic_p)
plt.title('Tuning AR term')
plt.xlabel('p (AR term)')
plt.ylabel('AIC/BIC score')
plt.legend(['AIC score','BIC score'])
plt.show()

    
# Tuning the MA Term #
aic_p = []
bic_p = []

p = range(1,6) # [1,2,3,4,5]

# aic/bic score for different values of p
for i in p:
    model = ARIMA(train_set, order=(i,1,0)) # define AR model
    model_fit = model.fit(disp=-1) # fit the model
    aic_temp = model_fit.aic # get aic score
    bic_temp = model_fit.bic # get bic score
    aic_p.append(aic_temp) # append aic score
    bic_p.append(bic_temp) # append bic score
    
# Plot of AIC/BIC score for AR term
plt.plot(range(1,6),aic_p, color='red')
plt.plot(range(1,6),bic_p)
plt.title('Tuning AR term')
plt.xlabel('p (AR term)')
plt.ylabel('AIC/BIC score')
plt.legend(['AIC score','BIC score'])
plt.show()  
   
# Fitting ARIMA model
model = ARIMA(train_set, order=(2,1,2))
model_fit_0 = model.fit()
model_fit_0

# Forecast using ARIMA model
past = train_set.tolist()
predictions = []
    
for i in range(len(test_set)):
    model = ARIMA(past, order=(2,1,2)) 
    model_fit = model.fit(disp=-1, start_params=model_fit_0.params)
    forecast_results = model_fit.forecast() 
        
    pred = forecast_results[0][0] 
    predictions.append(pred) 
    past.append(test_set[i]) 
    
# calculate mse
error = mean_squared_error(test_set, predictions)
print('Test MSE: {mse}'.format(mse=error))

# Plot forecasted and actual values
plt.plot(test_set)
plt.plot(test_set.index, predictions, color='red')
plt.title('Visa Stock Price')
plt.xlabel('Date')
plt.ylabel('Price in dollars')
plt.legend(['test_set','predictions'])
plt.show()