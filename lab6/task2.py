from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import re

train = pd.read_csv('titanic_train.csv')
x_train = train.drop('survived', axis=1)
y_train = train['survived']
x_test = pd.read_csv('titanic_reserved.csv')

cols_to_delete = ['ticket']
for col in x_train.columns:
    if (x_train[col].isna().sum() / len(x_train) > 1/3): cols_to_delete.append(col)
x_train.drop(cols_to_delete, axis=1, inplace=True)
x_test.drop(cols_to_delete, axis=1, inplace=True)

replacements = {
    'Mr': ['Rev', 'Col', 'Dr', 'Major', 'Don', 'Capt'],
    'Mrs': ['Dona', 'Countess'],
    'Miss': ['Mlle', 'Ms']
}

def make_honorific(x):
    honorific = re.search(r' (\S+?)\. ', x).group(1)    

    for k in replacements:
        if (honorific in replacements[k]): return k
    
    return honorific

x_train['honorific'] = x_train['name'].apply(make_honorific)
x_test['honorific'] = x_test['name'].apply(make_honorific)

x_train['age'] = x_train.groupby('honorific')['age'].transform(lambda x: x.fillna(x.mean()))
x_test['age'] = x_test.groupby('honorific')['age'].transform(lambda x: x.fillna(x.mean()))

x_train.drop(['honorific', 'name'], axis=1, inplace=True)
x_test.drop(['honorific', 'name'], axis=1, inplace=True)

dummies = pd.get_dummies(x_train.select_dtypes(exclude=np.number), prefix=x_train.select_dtypes(exclude=np.number).columns, drop_first=True)
x_train[dummies.columns] = dummies
x_train.drop(x_train.select_dtypes(exclude=np.number).columns, axis=1, inplace=True)
dummies = pd.get_dummies(x_test.select_dtypes(exclude=np.number), prefix=x_test.select_dtypes(exclude=np.number).columns, drop_first=True)
x_test[dummies.columns] = dummies
x_test.drop(x_test.select_dtypes(exclude=np.number).columns, axis=1, inplace=True)

lre = LogisticRegression(max_iter=1000)
lre.fit(x_train, y_train)
y_pred = lre.predict(x_test)

print(list(y_pred))