import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

df = pd.read_csv('persons_pics_train.csv')

print('1.1:', len(df['label'].unique()))
print('1.2:', df['label'].value_counts()['Gerhard Schroeder'] / df['label'].value_counts().sum())

mean_persons = df.groupby('label').mean()
print('1.3:', mean_persons.loc['Junichiro Koizumi'][0])

# Задание 1.4, откомментируйте, если надо
# fig, axs = plt.subplots(3, 4)

# for k, label in enumerate(df['label'].unique()):
#     axs[k//4][k%4].set_title(label)

#     img = []
#     for i in range(62):
#         img.append([])
#         for j in range(47):
#             img[i].append(mean_persons.loc[label][i*47+j])

#     axs[k//4][k%4].imshow(img, cmap='gray', vmin=0, vmax=1)

# plt.show()

ab = mean_persons.loc[['Jacques Chirac', 'Junichiro Koizumi']].prod(axis=0).sum()
a = sqrt(mean_persons.loc[['Jacques Chirac', 'Jacques Chirac']].prod(axis=0).sum())
b = sqrt(mean_persons.loc[['Junichiro Koizumi', 'Junichiro Koizumi']].prod(axis=0).sum())
print('1.5:', ab / (a * b))

x = df.drop('label', axis=1)
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11, stratify=y)
svc = SVC(kernel='linear', random_state=11)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print('2.1:', f1_score(y_test, y_pred, average='weighted'))

tuned_parameters = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000], 'class_weight': [None, 'balanced'], 'random_state':[11]}]


# cv = GridSearchCV(SVC(), tuned_parameters, refit=True, verbose=3)
# cv.fit(x_train, y_train)
# print('2.2-2.4:', cv.best_params_)
# y_pred = cv.predict(x_test)
# print('2.5:', f1_score(y_test, y_pred, average='weighted'))

# Бинарный поиск ищет кол-во компонент
l = 0
r = min(2914, len(x_train)) + 1

while (r - l > 1):
    m = (r + l) // 2
    pca = PCA(n_components=m, svd_solver='full')
    pca.fit(x_train)
    e = sum(pca.explained_variance_ratio_)

    if (e >= 0.95): r = m
    else: l = m

print('2.6:', m)

pca = PCA(n_components=m, svd_solver='full')
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
cv = GridSearchCV(SVC(), tuned_parameters, refit=True, verbose=3)
cv.fit(x_train, y_train)
print('2.7-2.9:', cv.best_params_)
y_pred = cv.predict(x_test)
print('2.10:', f1_score(y_test, y_pred, average='weighted'))