# Creating training and the testing set
train_data = final_data.loc['2012-12-31' : '2014-12-18']

test_data = final_data.loc['2014-12-19': ]


# First five records of the train data
train_data.head()

# Let us define the model variable
model = mSSA(rank = 20)

# Updating the model
model.update_model(train_data)

# Define the prediction dataframe
df = model.predict('MT_020', '2014-12-19 01:00:00', '2014-12-20  00:00:00')

# Set the figure size
plt.figure(figsize = (16, 6))

plt.plot( df['Mean Predictions'], label = 'predictions')

plt.fill_between(df.index, df['Lower Bound'], df['Upper Bound'], alpha = 0.1)

plt.plot(test_data['MT_020'].iloc[:len(df['Mean Predictions'])], label = 'Actual', alpha = 1.0)

# Set the title of the plot 
plt.title('Forecasting 1 day ahead')

plt.legend()

plt.show()

# Define the prediction dataframe
df = model.predict('MT_020', '2014-12-19 01:00:00', '2014-12-20  00:00:00', confidence = 99.9)

# Set the figure size
plt.figure(figsize = (16, 6))

plt.plot( df['Mean Predictions'], label = 'predictions')

plt.fill_between(df.index, df['Lower Bound'], df['Upper Bound'], alpha = 0.1)

plt.plot(test_data['MT_020'].iloc[:len(df['Mean Predictions'])], label = 'Actual', alpha = 1.0)

# Set the title of the plot
plt.title('Forecasting 1 day ahead')

plt.legend()

plt.show()