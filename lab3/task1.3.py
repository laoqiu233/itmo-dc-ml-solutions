from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('data2.csv')
test = pd.read_csv('data3.csv')

def clean_data(df):    
    df.replace('?', np.nan, inplace=True)

    if 'label' in df.columns:
        for col in df.columns:
            df[col] = df.groupby("label")[col].transform(lambda x: x.fillna(x.mode()[0]))
    else:
        for col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return pd.get_dummies(
        df,
        columns=df.select_dtypes(include=[object]).columns,
        drop_first = True
    )


train = clean_data(train)
test = clean_data(test)
test = test.reindex(columns=train.columns, fill_value=0)

x_train = train.drop('label', axis=1)
y_train = train['label']
x_test = test.reindex(columns=x_train.columns, fill_value=0)

scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

THRESHOLD = 0.05
corrs = [(y_train.corr(x_train[col]), col) for col in x_train.columns]
to_delete = [col for corr, col in corrs if corr < THRESHOLD]
x_train.drop(to_delete, axis=1, inplace=True)
x_test.drop(to_delete, axis=1, inplace=True)

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 25)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5)

knn_gscv.fit(x_train, y_train)
y_train_pred = knn_gscv.predict(x_train)
print('f1 score:', f1_score(y_train, y_train_pred))
y_pred = knn_gscv.predict(x_test)

with open('result', 'w') as file:
    file.write(str(list(y_pred)))

print(knn_gscv.best_score_)
print(knn_gscv.best_params_)