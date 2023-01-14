### Variables
training_start_date ='1982-09-27'
training_end_date ='2007-09-17'
testing_start_date = '2007-09-17'

plot_start_date = '2007-09-17'
plot_end_date = '2009-12-31'

pred_start_date = '2007-09-17'
pred_end_date = '2008-10-17'

##### Federal Funds Target Rate

### format data
data = pd.read_csv('./index.csv').fillna(method='ffill')
# data_rm_lit = data.iloc[:,:4]
# data_rmna = data_rm_lit.dropna()
data_rmna = data
ymd = pd.DataFrame({'Year': data_rmna['Year'],'Month': data_rmna['Month'],'Day':data_rmna['Day']})
time = pd.to_datetime(ymd)
testdf = pd.DataFrame({'time':time, 'Federal Funds Target Rate': data_rmna['Federal Funds Target Rate']})
final_data =  pd.pivot_table(testdf,index=["time"])

### Training Model 

train_data = final_data.loc[training_start_date:training_end_date]
test_data = final_data.loc[testing_start_date:]

FFTRmodel = mSSA(rank=1)
FFTRmodel.update_model(train_data)
pred = FFTRmodel.predict('Federal Funds Target Rate',pred_start_date,pred_end_date)

### Plotting
plt.figure(figsize = (16, 6))

plt.plot(pred['Mean Predictions'], label = 'predictions')

plt.fill_between(pred.index, pred['Lower Bound'], pred['Upper Bound'], alpha = 0.1)

plt.plot(final_data['Federal Funds Target Rate'].loc[plot_start_date:plot_end_date], label = 'Actual', alpha = 1.0)

# Set the title of the plot 
plt.title('Forecasting 1 day ahead')

plt.legend()

plt.show()


##### Unemployment Rate

### format data
data = pd.read_csv('./index.csv').fillna(method='ffill')
# data_rm_lit = data.iloc[:,:4]
# data_rmna = data_rm_lit.dropna()
data_rmna = data
ymd = pd.DataFrame({'Year': data_rmna['Year'],'Month': data_rmna['Month'],'Day':data_rmna['Day']})
time = pd.to_datetime(ymd)
testdf = pd.DataFrame({'time':time, 'Unemployment Rate': data_rmna['Unemployment Rate']})
final_data =  pd.pivot_table(testdf,index=["time"])

### Training Model 

train_data = final_data.loc[training_start_date:training_end_date]
test_data = final_data.loc[testing_start_date:]

UnemploymentRateModel = mSSA(rank=1)
UnemploymentRateModel.update_model(train_data)
pred = UnemploymentRateModel.predict('Unemployment Rate',pred_start_date,pred_end_date)

### Plotting
plt.figure(figsize = (16, 6))

plt.plot(pred['Mean Predictions'], label = 'predictions')

plt.fill_between(pred.index, pred['Lower Bound'], pred['Upper Bound'], alpha = 0.1)

plt.plot(final_data['Unemployment Rate'].loc[plot_start_date:plot_end_date], label = 'Actual', alpha = 1.0)

# Set the title of the plot 
plt.title('Forecasting Unemployment Rate 1 Month ahead')

plt.legend()

plt.show()


##### Effective Federal Funds Rate

### format data
data = pd.read_csv('./index.csv').fillna(method='ffill')
# data_rm_lit = data.iloc[:,:4]
# data_rmna = data_rm_lit.dropna()

data_rmna = data
ymd = pd.DataFrame({'Year': data_rmna['Year'],'Month': data_rmna['Month'],'Day':data_rmna['Day']})
time = pd.to_datetime(ymd)
testdf = pd.DataFrame({'time':time, 'Effective Federal Funds Rate': data_rmna['Effective Federal Funds Rate']})
final_data =  pd.pivot_table(testdf,index=["time"])

### Training Model 

train_data = final_data.loc[training_start_date:training_end_date]
test_data = final_data.loc[testing_start_date:]

EffectiveFederalFundsRateModel = mSSA(rank=1)
EffectiveFederalFundsRateModel.update_model(train_data)
pred = EffectiveFederalFundsRateModel.predict('Effective Federal Funds Rate',pred_start_date,pred_end_date)

### Plotting
plt.figure(figsize = (16, 6))

plt.plot(pred['Mean Predictions'], label = 'predictions')

plt.fill_between(pred.index, pred['Lower Bound'], pred['Upper Bound'], alpha = 0.1)

plt.plot(final_data['Effective Federal Funds Rate'].loc[plot_start_date:plot_end_date], label = 'Actual', alpha = 1.0)

# Set the title of the plot 
plt.title('Forecasting Effective Federal Funds Rate Rate 1 Month ahead')

plt.legend()

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
