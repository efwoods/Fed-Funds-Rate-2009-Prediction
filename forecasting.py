# Installing the mSSA library
from mssa.mssa import mSSA
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# An advanced library for data visualization in python
import seaborn as sns

# A simple library for data visualization in python
import matplotlib.pyplot as plt

# To ignore warnings in the code
import warnings
warnings.filterwarnings('ignore')

### Variables
training_start_date ='1982-09-27'
training_end_date ='2007-09-17'
testing_start_date = '2007-09-17'

plot_start_date = '2007-09-17'
plot_end_date = '2009-12-31'

pred_start_date = '2007-09-17'
pred_end_date = '2007-10-17'

##### Federal Funds Target Rate

### format data
data = pd.read_csv('./index.csv').fillna(method='ffill')
data = data.fillna(0)

# data_rm_lit = data.iloc[:,:4]
# data_rmna = data_rm_lit.dropna()


ymd = pd.DataFrame({'Year': data['Year'],'Month': data['Month'],'Day':data['Day']})
time = pd.to_datetime(ymd)
testdf = pd.DataFrame({
    'time':time,
    'Federal Funds Target Rate': data['Federal Funds Target Rate'],
    'Federal Funds Upper Target': data['Federal Funds Upper Target'],
    'Federal Funds Lower Target': data['Federal Funds Lower Target'],
    'Effective Federal Funds Rate': data['Effective Federal Funds Rate'],
    'Real GDP (Percent Change)': data['Real GDP (Percent Change)'],
    'Unemployment Rate': data['Unemployment Rate'],
    'Inflation Rate': data['Inflation Rate'],
    })
final_data =  pd.pivot_table(testdf,index=["time"])


### Training Model 

train_data = final_data.loc[training_start_date:training_end_date]
test_data = final_data.loc[testing_start_date:]

model = mSSA()
model.update_model(train_data)

# Creating Predictions with the model

predFederalFundsTargetRate = model.predict('Federal Funds Target Rate',pred_start_date,pred_end_date)

predUnemploymentRate = model.predict('Unemployment Rate',pred_start_date,pred_end_date)

### Creating a legend
legend_list = data.columns.to_list()
legend_list = legend_list[3:]
legend_prediction = ['Federal Funds Target Rate Prediction', 'Federal Funds Upper Target Prediction', 'Federal Funds Lower Target Prediction', 'Effective Federal Funds Rate Prediction', 'Real GDP (Percent Change) Prediction', 'Unemployment Rate Prediction', 'Inflation Rate Prediction']


### Plotting
plt.figure(figsize = (16, 6))

## Actual Values

FederalFundsTargetRateActual = plt.plot(final_data['Federal Funds Target Rate'].loc[plot_start_date:plot_end_date], 'r', label = 'FederalFundsTargetRate Actual', alpha = 1.0)

EffectiveFederalFundsRateActual = plt.plot(final_data['Effective Federal Funds Rate'].loc[plot_start_date:plot_end_date],'g', label = 'Effective Federal Funds Rate Actual', alpha = 1.0)

RealGDPPercentChangeActual = plt.plot(final_data['Real GDP (Percent Change)'].loc[plot_start_date:plot_end_date],'b', label = 'Real GDP (Percent Change) Actual', alpha = 1.0)

UnemploymentRateActual = plt.plot(final_data['Unemployment Rate'].loc[plot_start_date:plot_end_date],'y', label = 'Unemployment Rate Actual', alpha = 1.0)

InflationRateActual = plt.plot(final_data['Inflation Rate'].loc[plot_start_date:plot_end_date],'orange', label = 'Inflation Rate Actual', alpha = 1.0)

# Set the title of the plot 
plt.title('What would have happened in 2009 if the Fed had raised rates instead of lowering them?')

# Set legend
# plt.axis([FederalFundsTargetRatePred,FederalFundsTargetRateActual, UnemploymentRatePrediction, UnemploymentRateActual], label='Inline label')
plt.legend(loc="lower right")

plt.show()

'''

## Predictions

# Federal Funds Target Rate
FederalFundsTargetRatePred = plt.plot(predFederalFundsTargetRate['Mean Predictions'], 'g--', label = 'Federal FundsTarget Rate predictions')

plt.fill_between(predFederalFundsTargetRate.index, predFederalFundsTargetRate['Lower Bound'], predFederalFundsTargetRate['Upper Bound'], alpha = 0.1)


# Unemployment Rate
UnemploymentRatePrediction = plt.plot(predUnemploymentRate['Mean Predictions'],'r--', label = 'UnemploymentRate predictions')

plt.fill_between(predUnemploymentRate.index, predUnemploymentRate['Lower Bound'], predUnemploymentRate['Upper Bound'], alpha = 0.1)
'''

'''
### Forecast Monthly data
# Initialize prediction array
predictions = np.zeros((len(test_data.columns), 24*7))

upper_bound = np.zeros((len(test_data.columns), 24*7))

lower_bound = np.zeros((len(test_data.columns), 24*7))

actual = test_data.values[ :24 * 7, : ]

# Specify start time
start_time = pd.Timestamp('2014-12-19 01:00:00')

# Predict for seven days
days = 7

for day in range(days):
    
    # Get the final timestamp in the day
    end_time = start_time + pd.Timedelta(hours = 23)
    
    # Convert timestamps to string
    start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    
    end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Predict for each house
    for i, column in enumerate(test_data.columns):
        
        # Let us create the forecast
        df_30 = UnemploymentRateModel.predict(column, start_str, end_str)
        
        predictions[i, day * 24 : (day + 1) * 24] = df_30['Mean Predictions']
        
        upper_bound[i, day * 24 : (day + 1) * 24] = df_30['Upper Bound']
        
        lower_bound[i, day * 24 : (day + 1) * 24] = df_30['Lower Bound']
    
    # Fit the model with the already predicted values 
    df_insert = test_data.iloc[day * 24 : 24 * (day + 1), : ]
    
    # Update the model
    UnemploymentRateModel.update_model(df_insert)
    
    # Update start_time
    start_time = start_time + pd.Timedelta(hours = 24)
'''