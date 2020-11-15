import numpy as np
from StructuralPerturbationMethod import SPM
import networkx as nx
import matplotlib.pyplot as plt


def random_network_test(net):
    networks = {
        'ER': nx.erdos_renyi_graph(500,0.2),
        'WS': nx.watts_strogatz_graph(500, 8, 0.2),
        'BA': nx.barabasi_albert_graph(500, 5),
        'C': nx.complete_graph(500),
        'CB': nx.complete_bipartite_graph(150, 450)
    }

    print(f"Network {net}")
    G = networks[net]

    adj_matrix = np.array(nx.adjacency_matrix(G).todense())

    spm = SPM(adj_matrix, 0.1)
    #missing_links, preds = spm.run(verbose=False)
    missing_links, preds = spm.k_runs(k=10, verbose=False)
    # Draw graph
    def draw_graph_links(links):
        plt.figure(figsize=(10, 10))

        pos = nx.spring_layout(G)

        edge_list = links
        nodes_list = set([x for y in edge_list for x in y])

        # Draw nodes and edges not in edge_list
        #nx.draw_networkx_nodes(G, pos, nodelist=set(G.nodes) - nodes_list)
        #nx.draw_networkx_edges(G, pos, edgelist=set(G.edges) - set(edge_list), edge_color='black',
        #                       connectionstyle='arc3, rad = 0.3')

        # Draw nodes and edges in edge_list
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_list, node_color='r')
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color='r', arrowsize=15,
                               connectionstyle='arc3, rad = 0.5')

        nx.draw_networkx_labels(G, pos)
        plt.show()

    #draw_graph_links(missing_links)
    #draw_graph_links(preds)

def paper_network_test():
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

"""
test_dict = {
    'random': random_network_test,
    'paper': paper_network_test

}
"""

#test_dict['random']('ER')
#paper_network_test()
#random_network_test('ER')



if __name__ == "__main__":
    test_dict = {
    'random': random_network_test,
    'paper': paper_network_test
    }
    test_dict['random']('ER')