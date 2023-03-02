import numpy as np
import pygmtools as pygm

pygm.BACKEND = 'numpy'

# Generate a batch of graphs
batch_size = 10
A1 = np.random.rand(batch_size, 4, 4)
A2 = np.random.rand(batch_size, 4, 4)
n1 = n2 = np.repeat([4], batch_size)

# Build affinity matrix by the default inner-product function
conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, ne1, n2, ne2)

# Build affinity matrix by gaussian kernel
import functools

gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.)
K2 = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, ne1, n2, ne2, edge_aff_fn=gaussian_aff)

# Build affinity matrix based on node features
F1 = np.random.rand(batch_size, 4, 10)
F2 = np.random.rand(batch_size, 4, 10)
K3 = pygm.utils.build_aff_mat(F1, edge1, conn1, F2, edge2, conn2, n1, ne1, n2, ne2, edge_aff_fn=gaussian_aff)

# The affinity matrices K, K2, K3 can be further processed by GM solvers

X = pygm.rrwm(K3, n1, n2)
X = pygm.sinkhorn(X)
pass
