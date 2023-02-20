import numpy as np
from PIL import Image

# Менять только название файла, катинка сама появится
loadings = np.loadtxt('X_loadings_492.csv', delimiter=';')
reduced = np.loadtxt('X_reduced_492.csv', delimiter=';')

x = loadings @ reduced.transpose()

with Image.new(mode='L', size=x.shape) as img:
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            img.putpixel((i, j), int(x[i,j] * 255))

    img.show()