import numpy as np
import matplotlib.pyplot as plt

f = np.genfromtxt('13_36.csv', delimiter=',')
p_means = np.mean(f, axis=0)
for i in range(len(f)): f[i] -= p_means

theta = f.transpose() @ f
eig_values, eig_vectors = np.linalg.eig(theta)
idx = eig_values.argsort()[::-1]
eig_values = eig_values[idx]
eig_vectors = eig_vectors[:, idx[:2]]

for i in range(len(eig_values)):
    print(f"With {i+1} values: {sum(eig_values[0:i+1])/sum(eig_values)}")

z = f @ eig_vectors

print(z[0])

plt.scatter(x=z[:,0], y=z[:,1])
plt.show()