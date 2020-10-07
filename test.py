import numpy as np
from StructuralPerturbationMethod import SPM
import networkx as nx

"""
# Example
#G = nx.erdos_renyi_graph(50,0.2)
G = nx.watts_strogatz_graph(500, 8, 0.2)

adj_matrix = np.array(nx.adjacency_matrix(G).todense())

spm = SPM(adj_matrix, 0.1)
spm.run(verbose=False)
"""


# Paper example: results are zero-based, so they are scaled by -1
N = 9
A = np.zeros((N,N), dtype=np.int)


def get_adj_matrix(N, edges):
    """
    N: number of nodes
    edges: List[(tuples)]
    """

    A = np.zeros((N, N), dtype=np.int)
    for e in edges:
        A[e[0] - 1, e[1] - 1] = 1
        A[e[1] - 1, e[0] - 1] = 1
    return A

edges = [
        (1,2),
        (2,3),
        (4,5),
        (5,6),
        (7,8),
        (8,9),
        (1,4),
        (2,5),
        (3,6),
        (4,7),
        (5,8),
        (6,9),
        ]

A = get_adj_matrix(N=9, edges=edges)
delta_A = get_adj_matrix(9, [(5,8), (6,9)])

spm = SPM(A, None, delta_A, 2)
spm.run()

