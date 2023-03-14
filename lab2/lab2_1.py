from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
print('Mean X:', df['X'].mean())
print('Mean Y:', df['Y'].mean())

x = np.array(df['X']).reshape((len(df['X']), 1))
y = np.array(df['Y']).reshape((len(df['Y']), 1))

l = LinearRegression()
l.fit(x, y)
print('Коэффициент тета_1:', l.coef_)
print('Коэффициент тета_0:', l.intercept_)
print('R^2 статистика:', l.score(x, y))