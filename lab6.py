import numpy as np
import cv2
from matplotlib import pyplot as plt


#Переворачиватель картинок

I = cv2.imread('dolphin.jpg')[:, :, ::-1]
plt.figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(I)
plt.show()

I_ = np.transpose(I, (1, 0, 2))

plt.figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(I_)
plt.show()


#Умножение матриц (не скалярное )

A = np.array([[-1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
B = np.array([[1., -2., -3.], [7., 8., 9.], [4., 5., 6.], ])

C = A + B
D = A - B
E = A * B
F = A / B
G = A ** B

print('+\n', C, '\n')
print('-\n', D, '\n')
print('*\n', E, '\n')
print('/\n', F, '\n')
print('**\n', G, '\n')

#Формула среднеквадратичной ошибки

predictions=np.array([1,1,1])
labels=np.array([1,2,3])

error = (1/3)*np.sum(np.square(predictions-labels))

print(error)