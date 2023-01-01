import numpy as np
import pandas as pd
import pickle

toyota = pd.read_csv('toyota.csv')
x = toyota.iloc[:, 1:3]

y = toyota.iloc[:, 3]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

#pickle.dump(regressor, open('model.pkl','wb'))
#model = pickle.load(open('model.pkl','rb'))
