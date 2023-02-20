from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

f = np.loadtxt('34_16.csv', delimiter=',')
f_mean_cols = np.mean(f, axis=0)
for i in range(len(f)): f[i] -= f_mean_cols

# Сначало поставьте два, чтобы получить ответ на все пункты кроме 4
# потом постепенно изменять, пока не получите нужную дисперсию в 4
pca = PCA(n_components=2, svd_solver='full')
z = pca.fit_transform(f)
print('Доля объясненной дисперсии: ', sum(pca.explained_variance_ratio_))

print('Первый элемент:')
print(z[0])

# Посчитать, сколько пучков получиться на графике
plt.scatter(x=z[:,0], y=z[:,1])
plt.show()