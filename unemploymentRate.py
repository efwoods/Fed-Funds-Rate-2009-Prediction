


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