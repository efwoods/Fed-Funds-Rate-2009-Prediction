
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

