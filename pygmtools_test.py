import numpy as np
import pygmtools as pygm

pygm.BACKEND = 'numpy'
np.random.seed(1)

batch_size = 3
X_gt = np.zeros((batch_size, 4, 4))
X_gt[:, np.arange(0, 4, dtype=np.int64), np.random.permutation(4)] = 1
A1 = np.random.rand(batch_size, 4, 4)
A2 = np.matmul(np.matmul(X_gt.transpose((0, 2, 1)), A1), X_gt)
n1 = n2 = np.repeat([4], batch_size)

conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
import functools

gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.)  # set affinity function
K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, ne1, n2, ne2, edge_aff_fn=gaussian_aff)

X = pygm.rrwm(K, n1, n2, beta=100)
X = pygm.hungarian(X)
print(X)
print((X * X_gt).sum() / X_gt.sum())
