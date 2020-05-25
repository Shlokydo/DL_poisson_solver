import numpy as np
import scipy as sp

def get_aux(dim, alpha, smallest_coef, largest_coef):
    
    nnz = max(min(int(dim * dim * (1 - alpha) * 2), dim * dim), 0)

    row = np.random.randint(low=1, high=dim, size=nnz)
    col = np.random.randint(low=0, high=dim-1, size=nnz)

    data = np.random.rand(nnz)
    data[data < smallest_coef] = smallest_coef
    data[data > largest_coef] = largest_coef
    data = -data

    # duplicate (i,j) entries will be summed together
    return sp.sparse.coo_matrix((data, (row, col)), shape=(dim, dim))

def make_sparse_spd_matrix(dim=1, alpha=0.95, norm_diag=True, smallest_coef=.1, largest_coef=.9, format = 'coo'):
    
    chol = sp.sparse.eye(dim, format = 'coo')
    aux = get_aux(dim, alpha, smallest_coef, largest_coef)
    aux = sp.sparse.tril(aux, -1)

    chol += aux
    prec = chol.T.dot(chol)

    if norm_diag:
        
        d = prec.diagonal().reshape(1, prec.shape[0])
        d = 1. / np.sqrt(d)

        prec = prec.multiply(d)
        prec = prec.multiply(d.T)
    
    return prec.asformat(format)

def kron_spd_matrix(dim = 32, alpha = 0.95, smallest_coef=.1, largest_coef=.9, norm_diag = False, format = 'coo'):
  
  nnz = max(min(int(dim * dim * (1 - alpha) * 2), dim * dim), 0)

  row = np.random.randint(low = 0, high = dim, size = nnz)
  col = np.random.randint(low = 0, high = dim, size = nnz)

  data = np.random.rand(nnz)
  data[data < smallest_coef] = smallest_coef
  data[data > largest_coef] = largest_coef

  _A = sp.sparse.coo_matrix((data, (row, col)), shape=(dim, dim))
  _A = sp.sparse.tril(_A) + sp.sparse.tril(_A).T
  _D = sp.sparse.diags(_A.diagonal(), format = 'coo')
  _A -= _D
  _D = sp.sparse.diags(np.asarray(_A.sum(axis = 0)).squeeze(), format = 'coo')
  _A -= _D
  I = sp.sparse.eye(dim).tocoo()
  A = sp.sparse.kron(I , _A, 'coo') + sp.sparse.kron(_A, I, 'coo')

  return A.asformat(format)
