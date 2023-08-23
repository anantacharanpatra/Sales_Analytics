import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('advertising.csv')
x= df.drop(['Sales'], axis=1)
y = df['Sales']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)

import pickle
pickle.dump(lr, open('./model.pkl', 'wb'))
