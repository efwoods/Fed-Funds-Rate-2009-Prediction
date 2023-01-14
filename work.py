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

data = pd.read_csv('./index.csv').fillna(method='ffill')

from numpy import nan as NA
data_rm_lit = data.iloc[:,:4]
data_rmna = data_rm_lit.dropna()

data_rmna['Date'] = data_rmna['Year'].map(str)+"-"+data_rmna['Month'].map(str)+"-"+data_rmna['Day'].map(str)

data_rmna.head()

import seaborn as sns

sns.lineplot(x='Date',y='Federal Funds Target Rate',data=data_rmna)

import plotly.express as px

fig = px.line(data_rmna, x="Date", y="Federal Funds Target Rate", title='Federal Funds Target Rate Trend')
fig.show()