# import necessary librairies
import numpy as np

def PCA(df, k):
  x = np.array(df)

  # step 1: calculate the Mean normalization of x
  normalized_x = x - np.mean(x,axis=0)
  # step 2: calculate covariance matrix from the normalized_x
  cov = np.cov(normalized_x, rowvar=False)

  # make the parameter rowvar = False; check the documentation of np.cov to see why..

  # step 3: compute the eigen values and eigen vectors
  eig_val, eig_vec = np.linalg.eig(cov)

  # step 4: sort the eigen values in "descending" order, then use this sorted indicies to sort the eigen vectors.
  ind_sort = np.argsort(eig_val)[::-1]
  eig_val_sort = eig_val[ind_sort]
  eig_vect_sort = eig_vec[ind_sort]
  # print(eig_vect_sort)

  # step 5: select k eigen vectors
  reduced_eigen_vec = eig_vect_sort[:,:k]
  
  # step 6: transform the data
  z = normalized_x @ reduced_eigen_vec

  return z
