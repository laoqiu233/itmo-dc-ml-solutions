from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd

train = pd.read_csv('persons_pics_train.csv')
x_train = train.drop('label', axis=1)
y_train = train['label']

x_test = pd.read_csv('persons_pics_reserved.csv')

tuned_parameters = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000], 'class_weight': [None, 'balanced']}]
cv = GridSearchCV(SVC(), tuned_parameters, refit=True, verbose=3)
cv.fit(x_train, y_train)
print(list(cv.predict(x_test)))