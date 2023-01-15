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


##### Federal Funds Target Rate

### format data
data = pd.read_csv('./index.csv').fillna(method='ffill')
data = data.ffill()

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

### Variables
training_start_date ='1982-09-27'
training_end_date ='2007-09-01'
testing_start_date = '2007-09-01'

plot_start_date = '2007-09-01'
plot_end_date = '2009-12-31'


pred_window_size = 2

incident = time.loc[time =='2007-09-01'].index
pred_start_date = str(time[incident])
pred_start_date = pred_start_date[6:16]

pred_end_date = str(time.loc[incident+pred_window_size])
pred_end_date = pred_end_date[6:16]

### Training Model 

train_data = final_data.loc[training_start_date:training_end_date]
test_data = final_data.loc[testing_start_date:]

model = mSSA()
model.update_model(train_data)

# Creating Predictions with the model

predFederalFundsTargetRate = model.predict('Federal Funds Target Rate',pred_start_date,pred_end_date)
predEffectiveFederalFundsRate = model.predict('Effective Federal Funds Rate',pred_start_date,pred_end_date)
predRealGDPPercentChange = model.predict('Real GDP (Percent Change)', pred_start_date, pred_end_date)
predUnemploymentRate = model.predict('Unemployment Rate',pred_start_date,pred_end_date)
predInflationRate = model.predict('Inflation Rate', pred_start_date, pred_end_date)

### Plotting
plt.figure(figsize = (16, 6))

## Actual Values

FederalFundsTargetRateActual = plt.plot(final_data['Federal Funds Target Rate'].loc[plot_start_date:plot_end_date], 'r', label = 'Historical Federal Funds Target Rate', alpha = 1.0)

EffectiveFederalFundsRateActual = plt.plot(final_data['Effective Federal Funds Rate'].loc[plot_start_date:plot_end_date],'g', label = 'Historical Effective Federal Funds Rate', alpha = 1.0)

RealGDPPercentChangeActual = plt.plot(final_data['Real GDP (Percent Change)'].loc[plot_start_date:plot_end_date],'b', label = 'Historical Real GDP (Percent Change)', alpha = 1.0)

UnemploymentRateActual = plt.plot(final_data['Unemployment Rate'].loc[plot_start_date:plot_end_date],'c', label = 'Historical Unemployment Rate', alpha = 1.0)

InflationRateActual = plt.plot(final_data['Inflation Rate'].loc[plot_start_date:plot_end_date],'m', label = 'Historical Inflation Rate', alpha = 1.0)

## Predictions

# Federal Funds Target Rate
FederalFundsTargetRatePred = plt.plot(predFederalFundsTargetRate['Mean Predictions'], 'r:', label = 'Predicted Federal FundsTarget Rate')

plt.fill_between(predFederalFundsTargetRate.index, predFederalFundsTargetRate['Lower Bound'], predFederalFundsTargetRate['Upper Bound'], alpha = 0.1)

# Effective Federal Funds Rate
EffectiveFederalFundsRatePrediction = plt.plot(predEffectiveFederalFundsRate['Mean Predictions'],'g:', label = 'Predicted Effective Federal Funds Rate')

plt.fill_between(predEffectiveFederalFundsRate.index, predEffectiveFederalFundsRate['Lower Bound'], predEffectiveFederalFundsRate['Upper Bound'], alpha = 0.1)

# Real GDP (Percent Change)
RealGDPPercentChangePrediction = plt.plot(predRealGDPPercentChange['Mean Predictions'],'b:', label = 'Predicted Real GDP (Percent Change)')

plt.fill_between(predRealGDPPercentChange.index, predRealGDPPercentChange['Lower Bound'], predRealGDPPercentChange['Upper Bound'], alpha = 0.1)

# Unemployment Rate
UnemploymentRatePrediction = plt.plot(predUnemploymentRate['Mean Predictions'],'c:', label = 'Predicted Unemployment Rate')

plt.fill_between(predUnemploymentRate.index, predUnemploymentRate['Lower Bound'], predUnemploymentRate['Upper Bound'], alpha = 0.1)

# Inflation Rate
InflationRatePrediction = plt.plot(predInflationRate['Mean Predictions'],'m:', label = 'Predicted Inflation Rate')

plt.fill_between(predInflationRate.index, predInflationRate['Lower Bound'], predInflationRate['Upper Bound'], alpha = 0.1)



# Set the title of the plot 
plt.title('What would have happened in 2009 if the Fed had raised rates instead of lowering them?')

# Set legend
# plt.axis([FederalFundsTargetRatePred,FederalFundsTargetRateActual, UnemploymentRatePrediction, UnemploymentRateActual], label='Inline label')
plt.legend(loc="lower right")

plt.show()

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


### Aggregating Data
## Restructuring the prediction
effRate = pd.DataFrame({"Effective Federal Funds Rate":predFederalFundsTargetRate[pred_start_date:pred_end_date]["Mean Predictions"]})
effRate = effRate.rename_axis(index="time")
tempFinalPred = pd.DataFrame({"Effective Federal Funds Rate":final_data[:pred_start_date]["Effective Federal Funds Rate"]})

tempFinalPred = tempFinalPred.append(effRate)
pred_start_date = pred_end_date

## Concatonate all the new predictions into a single dataframe
# pd.DataFrame({"Effective Federal Funds Rate1":tempFinalPred["Effective Federal Funds Rate"],"Effective Federal Funds Rate2":tempFinalPred["Effective Federal Funds Rate"]})

'''
### Adding Months

# import packages
import numpy as np
  
# adding months to a given date
print('old date is : ' + str(np.datetime64('2022-04')))
new_date = np.datetime64('2022-04') + np.timedelta64(5, 'M')
print('new date is : '+str(new_date))
'''

#'2007-09-01' == time index 776
#incident = time.loc[time =='2007-09-01'].index
# to iterate through the time: incident +1 -> n

effRate = pd.DataFrame({"Effective Federal Funds Rate":predFederalFundsTargetRate[time[]:pred_end_date]["Mean Predictions"]})

### Converting the datetime into a string to address the prediction
# dateString = str(time[incident])
# dateString = dateString[6:16]
# predFederalFundsTargetRate[dateString]