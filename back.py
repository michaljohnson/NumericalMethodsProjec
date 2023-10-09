import numpy as np
from ctscanner import CTScanner
from random import random
from math import pi

def shoot(scanner):
  indices, intensity, lengths = scanner.shootRays(random()*pi)
  return indices, intensity, lengths

def swap_row(A, b, index_1, index_2):
  for i in range(A.shape[1]):
    tmp = A[index_1][i]
    A[index_1][i] = A[index_2][i]
    A[index_2][i] = tmp
  tmp = b[index_1]
  b[index_1] = b[index_2]
  b[index_2] = tmp

def forward_reduce(A, b):
  for current_col in range(A.shape[1]):
    highest_value_index = np.argmax(np.abs(A[current_col:, current_col])) + current_col
    if current_col != highest_value_index:
      swap_row(A, b, current_col, highest_value_index)
    val_pivot = A[current_col][current_col]
    for reduction_row in range(current_col + 1, A.shape[0]):
      factor = A[reduction_row,current_col] / val_pivot
      # current_col is the column we are opperating but also the row with the pivot
      to_subtract_from_A = factor * A[current_col]
      to_subtract_from_b= factor * b[current_col]
      A[reduction_row] -= to_subtract_from_A
      b[reduction_row] -= to_subtract_from_b

def backwards_reduce(A, b):
  for current_col in range(A.shape[1]-1, -1, -1):
    val_pivot = A[current_col][current_col]
    for reduction_row in range(current_col):
      factor = A[reduction_row, current_col] / val_pivot
      # current_col is the column we are opperating but also the row with the pivot
      to_subtract_from_A = factor * A[current_col]
      to_subtract_from_b = factor * b[current_col]
      A[reduction_row] -= to_subtract_from_A
      b[reduction_row] -= to_subtract_from_b

def create_x(A,b):
  x = np.zeros(A.shape[1])
  for current_row in range(A.shape[1]):
    x[current_row] = b[current_row] / A[current_row][current_row]
  return x


def rowrank(matrix):
  b = np.zeros(matrix.shape[0])
  forward_reduce(matrix, b)
  rank = 0
  nonzero_val = False
  for current_row in range(matrix.shape[0]):
    for current_col in range(matrix.shape[1]):
      val = matrix[current_row][current_col]
      # define threshhold
      if np.abs(val) > 0.0000001:
        nonzero_val = True
    if(nonzero_val):
      rank += 1
      nonzero_val = False
  return rank

def setUpLinearSystem(scanner):
  A = np.zeros((scanner.resolution ** 2, scanner.resolution ** 2))
  b = np.zeros(scanner.resolution ** 2)
  current_row = 0
  while(scanner.resolution ** 2 > np.linalg.matrix_rank(A)):
    if current_row + scanner.resolution >= scanner.resolution ** 2:
      A.resize(A.shape[0]+scanner.resolution, scanner.resolution ** 2, refcheck= False)
      b.resize(b.shape[0]+scanner.resolution, refcheck= False)
    indices, intensity, lengths = shoot(scanner)
    for i in range(len(indices)):
      b[current_row] = intensity[i]
      for j_index, j in enumerate(indices[i]):
        A.itemset((current_row, j), lengths[i][j_index])
      current_row += 1
  return A, b

"""
# would setup a fullrank matrix, but we don't need a square matrix, this part has to be solved in c) :-(
# also this solution has a VERY BAD time complexity
def setUpLinearSystem(scanner):
  A = np.zeros([scanner.resolution ** 2, scanner.resolution ** 2])
  b = np.zeros(scanner.resolution ** 2)
  current_row = 0
  while(True):
    # fill the matrix
    indices, intensity, lengths = shoot(scanner)
    # iterate for each ray once (in this case 20 times)
    for i in range(len(indices)):
      rowrank = np.linalg.matrix_rank(A)
      print(rowrank)
      b[current_row] = intensity[i]
      # iterate over every index in a single ray
      for j_index, j in enumerate(indices):
          A[current_row][j] = lengths[j_index]
      # delete row if rowrank wasn't increased by adding this row
      if np.linalg.matrix_rank(A) == rowrank:
        b[current_row] = 0.0
        for j in range(scanner.resolution ** 2):
          A[current_row][j] = 0.0
      # don't get a out of array error
      else:
        current_row += 1
      if current_row == scanner.resolution ** 2:
        return A, b
      # if we succesfully added a row we can go to the next row
"""

def solveLinearSystem(A, b):
  forward_reduce(A,b)
  #A.resize(rowrank(A), A.shape[1]) possible if we don't assume square matrix
  backwards_reduce(A,b)
  x = create_x(A,b)
  return x


scanner = CTScanner()
setUpLinearSystem(scanner)
