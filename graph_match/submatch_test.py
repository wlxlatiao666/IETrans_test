import torch  # pytorch backend
import pygmtools as pygm
import matplotlib.pyplot as plt  # for plotting
from matplotlib.patches import ConnectionPatch  # for plotting matching result
import networkx as nx  # for plotting graphs

pygm.BACKEND = 'pytorch'  # set default backend for pygmtools
_ = torch.manual_seed(1)  # fix random seed

# generate the larger graph
num_nodes2 = 10
A2 = torch.rand(num_nodes2, num_nodes2)
A2 = (A2 + A2.t() > 1.) * (A2 + A2.t()) / 2
torch.diagonal(A2)[:] = 0
n2 = torch.tensor([num_nodes2])

# generate the smaller graph
num_nodes1 = 5
G2 = nx.from_numpy_array(A2.numpy())
pos2 = nx.spring_layout(G2)
pos2_t = torch.tensor([pos2[_] for _ in range(num_nodes2)])
selected = [0]  # build G1 as a cluster in visualization
unselected = list(range(1, num_nodes2))
while len(selected) < num_nodes1:
    dist = torch.sum(torch.sum(torch.abs(pos2_t[selected].unsqueeze(1) - pos2_t[unselected].unsqueeze(0)), dim=-1),
                     dim=0)
    select_id = unselected[torch.argmin(dist).item()]  # find the closest node from unselected
    selected.append(select_id)
    unselected.remove(select_id)
selected.sort()
A1 = A2[selected, :][:, selected]
X_gt = torch.eye(num_nodes2)[selected, :]
n1 = torch.tensor([num_nodes1])

# visualize the graphs
G1 = nx.from_numpy_array(A1.numpy())
pos1 = {_: pos2[selected[_]] for _ in range(num_nodes1)}
color1 = ['#FF5733' for _ in range(num_nodes1)]
color2 = ['#FF5733' if _ in selected else '#1f78b4' for _ in range(num_nodes2)]
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Subgraph 1')
plt.gca().margins(0.4)
nx.draw_networkx(G1, pos=pos1, node_color=color1)
plt.subplot(1, 2, 2)
plt.title('Graph 2')
nx.draw_networkx(G2, pos=pos2, node_color=color2)

# build affinity matrix
conn1, edge1 = pygm.utils.dense_to_sparse(A1)
conn2, edge2 = pygm.utils.dense_to_sparse(A2)
import functools

gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.001)  # set affinity function
K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)
#
plt.figure(figsize=(4, 4))
plt.title(f'Affinity Matrix (size: {K.shape[0]}$\\times${K.shape[1]})')
plt.imshow(K.numpy(), cmap='Blues')

X = pygm.rrwm(K, n1, n2)
# print(X.sum((0, 1)))

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('RRWM Soft Matching Matrix')
plt.imshow(X.numpy(), cmap='Blues')
plt.subplot(1, 2, 2)
plt.title('Ground Truth Matching Matrix')
plt.imshow(X_gt.numpy(), cmap='Blues')

X = pygm.hungarian(X)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title(f'RRWM Matching Matrix (acc={(X * X_gt).sum() / X_gt.sum():.2f})')
plt.imshow(X.numpy(), cmap='Blues')
plt.subplot(1, 2, 2)
plt.title('Ground Truth Matching Matrix')
plt.imshow(X_gt.numpy(), cmap='Blues')

plt.figure(figsize=(8, 4))
plt.suptitle(f'RRWM Matching Result (acc={(X * X_gt).sum() / X_gt.sum():.2f})')
ax1 = plt.subplot(1, 2, 1)
plt.title('Subgraph 1')
plt.gca().margins(0.4)
nx.draw_networkx(G1, pos=pos1, node_color=color1)
ax2 = plt.subplot(1, 2, 2)
plt.title('Graph 2')
nx.draw_networkx(G2, pos=pos2, node_color=color2)
for i in range(num_nodes1):
    j = torch.argmax(X[i]).item()
    con = ConnectionPatch(xyA=pos1[i], xyB=pos2[j], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="green" if X_gt[i, j] == 1 else "red")
    plt.gca().add_artist(con)
