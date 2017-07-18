from scipy import sparse as sp
import scipy
import numpy as np
import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering
import operator
import matplotlib.pyplot as plt

def permuteMatrix(perm, orig):
  permuted_m = []
  for r in perm:
    permuted_m.append(orig[:, r])
  return np.asarray(permuted_m)

# takes some ordering of indices and returns the corresponding
# permutation matrix (permuted identity matrix)
def generatePermutationMatrix(index_arr):
  n = len(index_arr)

  P = []
  for i in range(0,n):
    row = np.zeros(n)
    row[index_arr[i]] = 1.0
    P.append(row)

  return np.asarray(P) 

def mapListToString(arr):
  ret_str = ''
  for b in arr:
    ret_str += str(int(b))

  return ret_str

def getDuplicateColumnIndices(X):
  dataToIndex = {}
  for i in range(X.shape[1]):
    col = mapListToString(X[:,i])
    #print(col)
    if col in dataToIndex:
      dataToIndex[col].append(i)
    else:
      dataToIndex[col] = [i]

  return dataToIndex

# produces permutation matrix given the input matrix
def permuteKSparse(X_orig):
  #index_arr contains mapping from the column to its old index
  #index_arr[i] accesses the ith column of the permuted X
  #contained value is index of original column in X
  index_arr = [i for i in range(0, X_orig.shape[1])]
  X = np.copy(X_orig)

  j = 0
  colToIndices = getDuplicateColumnIndices(X)
  while j < X.shape[1]:
    ref = X[:,j]
    ref_key = mapListToString(ref)

    col_indices = sorted(colToIndices[ref_key])

    #this column is unique
    if len(col_indices) == 1:
      j += 1
    else:
      #multiple columns of this type
      for i in range(1,len(col_indices)):
        j += 1
        if j == X.shape[1]:
          break

        #there is no consecutive, duplicate column
        if np.array_equal(X[:,j], ref) == False:
          #perform the swap
          #j is the column we are going to swap out to column col_indices[i]
          #col_indices[i] is the column index we are going to swap in to j
          swap_in_col = col_indices[i]

          swap_out_key = mapListToString(X[:,j])

          #swap columns of X
          X[:,[j,swap_in_col]] = X[:,[swap_in_col,j]]

          #update our index_arr
          #swap out temp, swap in swap_in_col
          temp = index_arr[j]
          index_arr[j] = index_arr[swap_in_col]
          index_arr[swap_in_col] = temp

          #update column index mapping
          col_swap_out_index = colToIndices[swap_out_key].index(j)
          colToIndices[swap_out_key][col_swap_out_index] = swap_in_col
          col_indices[i] = j

      j += 1

  return generatePermutationMatrix(index_arr)

def permuteKSparse_kth(X_orig, KS, K_n):

  perm_factor = KS[K_n]

  st_row = 0
  for i in range(1,K_n):
    st_row += KS[i]

  end_row = st_row + KS[K_n]

  X_n = X_orig[st_row:end_row]

  return permuteKSparse(X_n)

def generateSplitsBetweenFactors(X_type_subset):
  X = np.copy(X_type_subset)
  splits = []
  for i in range(1, X.shape[1]):
    if np.array_equal(X[:,i], X[:,i-1]) == False:
      splits.append(i)

  return splits

#recursively permutes block diagonals by factor type
#currently a little buggy (doesn't work for more than 2 factor types)
def permuteBlocks(X_orig, KS, narrow_type_order, P_orig):
  X = np.copy(X_orig)
  P = np.copy(P_orig)

  curr_factor_type = narrow_type_order[0]

  perm_matrix = permuteKSparse_kth(X, KS, curr_factor_type)

  X = X.dot(perm_matrix.T)

  P = P.dot(perm_matrix.T)

  if len(narrow_type_order) == 1:
    return P

  st_row = 0
  for i in range(1, curr_factor_type):
    st_row += KS[i]

  end_row = st_row + KS[curr_factor_type]

  type_subset = X[st_row:end_row]

  splits = generateSplitsBetweenFactors(type_subset)

  recursed_perm = np.zeros((X.shape[1], X.shape[1]))
  
  j = 0
  for i in range(len(splits)+1):
    if i == len(splits):
      consec_block = X[:, splits[i-1]:]
      p_block = P.T[splits[i-1]:]
    elif i == 0:
      consec_block = X[:, 0:splits[i]]
      p_block = P.T[0:splits[i]]
    else:
      consec_block = X[:, splits[i-1]:splits[i]]
      p_block = P.T[splits[i-1]:splits[i]]

    block_perm = permuteBlocks(consec_block, KS, narrow_type_order[1:], p_block.T)

    for i in range(block_perm.shape[1]):
      recursed_perm[:, j] = block_perm[:, i]
      j += 1

  return np.array(recursed_perm).T




