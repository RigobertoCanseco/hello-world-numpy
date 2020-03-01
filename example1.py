"""
Vectors, matrices and arrays
"""
import numpy as np
from scipy import sparse

# create a vector as a row
vector_row = np.array([1, 2, 3])
print("vector_row")
print(vector_row)
# [1 2 3]

# create a vector as a column
vector_column = np.array([[1], [2], [3]])
print("vector_column")
print(vector_column)
'''
[[1]
 [2]
 [3]]
'''

# create a matrix
matrix = np.array([[1, 2], [3, 4], [5, 6]])
print("matrix")
print(matrix)
'''
[[1 2]
 [3 4]
 [5 6]]
'''

# matrix
matrix_object = np.array([[1, 2], [3, 4], [5, 6]])
print("matrix_object")
print(matrix_object)

# create compressed sparse row (CSR) matrix
matrix = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
matrix_sparse = sparse.csr_matrix(matrix)
print("matrix_sparse")
print(matrix_sparse)

# Selection
vector = np.array([1, 2, 3, 4, 5, 6])
matrix = np.array([[1, 2], [3, 4], [5, 6]])
print(vector[2])
print(matrix[2, 0])

print("[:] = ", vector[:])
print("[:3] = ", vector[:3])
print("[3:] = ", vector[3:])
print("[-1] = ", vector[-1])
print("[:2, :] = ", matrix[:2, :])
print("[:2, 1:] = ", matrix[:2, 1:])
print("[:2, -1] = ", matrix[:2, -1])

# description
matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
print("description")
print(matrix.shape, matrix.size, matrix.ndim)

# operations
matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
# create a function that adds 100 to something
add_100 = lambda i: i + 100
print(add_100)

# create a vectorized function
vectorized_function = np.vectorize(add_100)
print(vectorized_function)

# apply function to all elements in matrix
matrix = vectorized_function(matrix)
print("vectorized_function(matrix)")
print(matrix)

# max
print("max:", np.max(matrix))

# min
print("min:", np.min(matrix))

# mean
print("mean:", np.mean(matrix))

# variance
print("variance:", np.var(matrix))

# standard deviation
print("deviation:", np.std(matrix))

# reshaping arrays
matrix = np.array([[1, 2, 3], [4, 5, 6]])
matrix = matrix.reshape(3, 2)
'''
[[1 2]
 [3 4]
 [5 6]]
'''
print(matrix)

matrix = matrix.reshape(1, 6)
print(matrix)
# [[1 2 3 4 5 6]]

matrix = matrix.reshape(6, 1)
print(matrix)
'''
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]]
'''

matrix = matrix.reshape(2, -1)
'''
[[1 2 3]
 [4 5 6]]
'''
print(matrix)

matrix = matrix.reshape(6)
# [1 2 3 4 5 6]
print(matrix)

# transposing
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix = matrix.T
print(matrix)
'''
[[1 4 7]
 [2 5 8]
 [3 6 9]]
'''

# transpose vector
# [1 2 3 4 5 6]
print(np.array([1, 2, 3, 4, 5, 6]).T)

# transpose row vector
'''
[[1]
 [2]
 [3]
 [4]
 [5]
 [6]]
'''
print(np.array([[1, 2, 3, 4, 5, 6]]).T)

# flattening a matrix
matrix = np.array([[1, 2, 3, 4, 5, 6]]).T
print(matrix.flatten())

# rank of a matrix
matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
result = np.linalg.matrix_rank(matrix)
print(result)

# determinant
result = np.linalg.det(matrix)
print(result)

# diagonal
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

result = matrix.diagonal()
# [1 5 9]
print(result)

result = matrix.diagonal(offset=1)
# [2 6]
print(result)

result = matrix.diagonal(offset=-1)
# [4 8]
print(result)

# trace
result = matrix.trace()
# [4 8]
print(result)