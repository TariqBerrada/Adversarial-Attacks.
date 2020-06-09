import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import cv2

def rgb2gray(image):
    s = np.shape(image)
    gray = np.ones((s[0], s[1]))
    gray = (0.2989*image[:, :, 0] + 0.5870*image[:, :, 1] + 0.1140*image[:, :, 2])/255
    return gray

def SD(image, xi, n):
    neighborhood = image[xi[0]-n-1:xi[0]+n, xi[1]-n-1:xi[1]+1]
    mu = np.mean(neighborhood)
    SD = np.sqrt(np.sum((neighborhood - mu)**2)/(2*n+1)**2)
    return SD

def sen(image, xi, n):
    return 1/SD(image, xi, n)

def distance_metric(image, sensibility_matrix,  perturbations):
    return np.sum(np.multiply(sensibility_matrix, perturbations))

def get_n_biggest(mat, n):
    mat_shape = np.shape(mat)
    arr = np.reshape(mat, mat_shape[0]*mat_shape[1])
    ar_indexes = arr.argsort()[-n:][::-1]
    mat_indexes = []
    for i in ar_indexes:
        y = i%mat_shape[1] 
        x = i//mat_shape[1]
        mat_indexes.append((x, y))

    return mat_indexes
