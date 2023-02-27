import numpy as np
import pygmtools as pygm
pygm.BACKEND = 'numpy'
np.random.seed(1)

# Generate 10 isomorphic graphs
graph_num = 10
As, X_gt = pygm.utils.generate_isomorphic_graphs(node_num=4, graph_num=10)
As_1, As_2 = [], []
for i in range(graph_num):
    for j in range(graph_num):
        As_1.append(As[i])
        As_2.append(As[j])
As_1 = np.stack(As_1, axis=0)
As_2 = np.stack(As_2, axis=0)

# Build affinity matrix
conn1, edge1, ne1 = pygm.utils.dense_to_sparse(As_1)
conn2, edge2, ne2 = pygm.utils.dense_to_sparse(As_2)
import functools
gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, None, None, None, None, edge_aff_fn=gaussian_aff)
K = K.reshape(graph_num, graph_num, 4*4, 4*4)
print(K.shape)

# Solve the multi-matching problem
X = pygm.cao(K)
print(X)
print((X * X_gt).sum() / X_gt.sum())