from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Параметры
df = pd.read_csv('fish_train.csv')
test_size = 0.2
random_state=35

def get_splits():
    species = df['Species']
    x = df.drop(['Weight', 'Species'], axis=1)
    y = df['Weight']

    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=species)

# Построение базовой модели

x_train, x_test, y_train, y_test = get_splits()

print('Среднее колонки Width:', x_train['Width'].mean())

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print('R2 оценка для тестового набора:', r2_score(y_test, y_pred))

# Добавление предварительной обратботки признаков

# Скорее всего, тройка будет Length1...3, ну все равно будьте внимательны
print('Матрица корреляции:')

x_train, x_test, y_train, y_test = get_splits()

# print(df.drop('Species', axis=1).corr())
pca = PCA(n_components=1, svd_solver='full')
pca.fit(x_train[['Length1', 'Length2', 'Length3']])
print('Доля объясненной дисперсии:', pca.explained_variance_ratio_)
df['Lengths'] = pca.transform(df[['Length1', 'Length2', 'Length3']])
df.drop(['Length1', 'Length2', 'Length3'], axis=1, inplace=True)

x_train, x_test, y_train, y_test = get_splits()

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print('R2 оценка для тестового набора:', r2_score(y_test, y_pred))

# Модификация признаков

df[['Width', 'Height', 'Lengths']] = df[['Width', 'Height', 'Lengths']].apply(lambda x: x**3)

x_train, x_test, y_train, y_test = get_splits()

print('Среднее Width после возведения в куб:', x_train['Width'].mean())

unique_species = df['Species'].unique()
for i in unique_species:
    plt.scatter(df[df['Species'] == i]['Width'], df[df['Species'] == i]['Weight'], label=i)
plt.legend()
plt.xlabel('Width')
plt.ylabel('Weight')
plt.show()

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print('R2 оценка для тестового набора:', r2_score(y_test, y_pred))

dummies = pd.get_dummies(df['Species'])
df[list(dummies.columns)] = dummies
x_train, x_test, y_train, y_test = get_splits()
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print('R2 оценка для тестового набора:', r2_score(y_test, y_pred))
df.drop(list(dummies.columns), axis=1, inplace=True)

dummies = pd.get_dummies(df['Species'], drop_first=True)
df[list(dummies.columns)] = dummies
x_train, x_test, y_train, y_test = get_splits()
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print('R2 оценка для тестового набора:', r2_score(y_test, y_pred))