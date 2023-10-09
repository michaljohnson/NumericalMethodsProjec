import numpy as np
from ctscanner import CTScanner

# Task a)
# Implement a method, calculating the rowrank of a given matrix and return it.
# Obviously, you're not allowed to use any method solving that problem for you.

def rowrank(matrix):
  A = matrix
  b = np.zeros(matrix.shape[0])
  def swap_row(A, b, index_1, index_2):
    for i in range(A.shape[1]):
      tmp = A[index_1][i]
      A[index_1][i] = A[index_2][i]
      A[index_2][i] = tmp
    tmp = b[index_1]
    b[index_1] = b[index_2]
    b[index_2] = tmp
  for current_col in range(A.shape[1]):
    highest_value_index = np.argmax(np.abs(A[current_col:, current_col])) + current_col
    if current_col != highest_value_index:
      swap_row(A, b, current_col, highest_value_index)
    val_pivot = A[current_col][current_col]
    for reduction_row in range(current_col + 1, A.shape[0]):
      factor = A[reduction_row, current_col] / val_pivot
      # current_col is the column we are opperating but also the row with the pivot
      to_subtract_from_A = factor * A[current_col]
      to_subtract_from_b = factor * b[current_col]
      A[reduction_row] -= to_subtract_from_A
      b[reduction_row] -= to_subtract_from_b
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


# Task b)
# Implement a method setting up the linear system, as described in the exercise.
# Make use of the scanner.shootRays(angle) function.

def setUpLinearSystem(scanner):
  A = np.zeros((scanner.resolution ** 2, scanner.resolution ** 2))
  b = np.zeros(scanner.resolution ** 2)
  counter = 0
  while(np.linalg.matrix_rank(A) < A.shape[1]):
    if(counter * scanner.resolution >= A.shape[0]):
      A.resize(A.shape[0] + scanner.resolution, A.shape[1], refcheck=False)
      b.resize(b.shape[0] + scanner.resolution, refcheck=False)
    indices, intensities, lengths = scanner.shootRays(np.random.random() * np.pi)
    for i in range(len(indices)):
      A[counter * scanner.resolution + i, indices[i]] = lengths[i]
    b[counter * scanner.resolution: (counter + 1) * scanner.resolution] = intensities
    counter += 1
  return A, b

scanner = CTScanner()
setUpLinearSystem(scanner)

# Task c)
# Implement the gaussian elimination method, to solve the given system of linear equations
# Add full pivoting to increase accuracy and stability of the solution

def solveLinearSystem(A, b):

  def swapRows(i, index):
    tmp = np.copy(A[i])
    A[i] = np.copy(A[index])
    A[index] = np.copy(tmp)
    tmp = b[i]
    b[i] = b[index]
    b[index] = tmp

  def rowreduce():
    for i in range(A.shape[1]):
      index = np.argmax(np.abs(A[i: ,i])) + i
      if(index != i):
        swapRows(i, index)
      for j in range(i+ 1, A.shape[0]):
        factor = A[j,i] / A[i,i]
        A[j] -= factor * A[i]
        b[j] -= factor * b[i]


  def backsubstitute():
    x = np.zeros(A.shape[1])
    for i in reversed(range(A.shape[1])):
      pred = 0
      for j in reversed(range(i, A.shape[1])):
        pred += A[i,j] * x[j]
      x[i] = (b[i] - pred) /A[i,i]
    return x

  rowreduce()
  x = backsubstitute()
  return x

