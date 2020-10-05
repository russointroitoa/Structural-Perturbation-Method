import numpy as np
import math

np.random.seed(42)

class SPM(object):

    """
    Given a graph G = (V, E), we can construct the adjacency matrix A of the E links.
    We randomly select a fraction p of the links to constitute a perturbation set ΔE, while the rest of the links E − ΔE
    constitute the set E_r . Denote by A and ΔA the corresponding adjacency matrices; obviously, A = AR + ΔA.
    """

    def __init__(self, A, p):
        """
        :param A: original adjacency matrix that would be decomposed in A = A_r + delta_A
        :param p: fraction of links to constitute a perturbation set delta_E
        :return:
        """

        self.A = A
        self.p = p
        self.N = self.A.shape[0]
        self.n_removed_link = math.ceil(self.N * self.p)

        # Extract a fraction p of links from A to generate delta_A
        links = np.where(np.triu(self.A) == 1)    # Tuple (x, y) with x and y are lists of indices, we consider only
                                                    # the upper triangle of the matrix and reflect the symmetry of the
                                                    # matrix in self.create_adj_matrix()
        indices = np.random.randint(0, links[0].shape[0], self.n_removed_link)
        self.links = list(zip(links[0][indices], links[1][indices]))

        # Create delta_A
        self.delta_A = self.create_adj_matrix(self.N, self.links)

        # Create A_r = A - delta_A
        self.A_r = self.A - self.delta_A


    def create_adj_matrix(self, N, edges):
        """
        :param N: number of nodes in the network
        :param edges: edges of the network
        :return:
        """
        matrix = np.zeros((N, N), dtype=np.int)
        for e in edges:
            matrix[e[0], e[1]] = 1
            matrix[e[1], e[0]] = 1
        return A

    def compute_delta_lambda(self, v, delta_A):
        """
        v : eigenvectors
        delta_A : perturbation matrix
        """
        res = np.empty(v.shape[0])

        for k in range(v.shape[0]):
            num = v[:, k].T.dot(delta_A).dot(v[:, k])
            den = v[:, k].T.dot(v[:, k])
            res[k] = num / den

        return res

    def compute_perturbed_matrix(self, A_r, delta_A):

        lambda_k, x_k = np.linalg.eig(A_r)
        """
        w: eigenvalues of A_r
        v: right-eigenvectors of A_r
        """

        delta_lambda_k = self.compute_delta_lambda(x_k, delta_A)

        perturbed_A = np.zeros(A_r.shape)

        # perturbed_A = perturbed_A + ((lambda_k + delta_lambda_k)*(x_k.dot(x_k.T)))
        for k in range(A_r.shape[0]):
            perturbed_A += (lambda_k[k] + delta_lambda_k[k]) * np.dot(x_k[:, k][:, None], x_k[:, k][None, :])

        # Clear perturbed_A: check the existing link in A_r [i.e. in E_r] and set those positions = -np.inf in
        # perturbed_A such that these links don't occur in the ranking to extract the most probable missing link.
        # What we are doing is to consider the set U - E_r, where U is the universe of possible links given N nodes
        perturbed_A[np.where(A_r == 1)] = -np.inf

        # Moreover we set to -np.inf also the element (i,i) in the diagonal since we do not have self-links
        np.fill_diagonal(perturbed_A, -np.inf)

        return perturbed_A

    def extract_ranking(self, perturbed_A, at):
        # ranking = perturbed_A.ravel().argsort()[::-1]

        # Since the matrix is symmetric, we can consider the upper triangle and consider the first half of results
        # (the second half represents the element of the low triangle of the matrix, set to zero)

        up_matrix_ranking = np.triu(perturbed_A).ravel().argsort()[::-1]
        ranking = up_matrix_ranking[:len(up_matrix_ranking) // 2]
        ranking = [(int(x / self.N), x % self.N) for x in ranking][:at]
        return ranking

    def evaluate(self, predicted_links):
        # We need to replicate each link such that there are both (x,y) and (y, x)
        preds = set(predicted_links).union(set([(y,x) for (x,y) in predicted_links]))
        links = set(self.links).union(set([(y,x) for (x,y) in self.links]))

        n_missing_links = len(links)
        correct_predicted = len(set(preds).intersection(links))
        return correct_predicted / n_missing_links

    def run(self, verbose=True):
        self.perturbed_A = self.compute_perturbed_matrix(self.A_r, self.delta_A)
        ranks = self.extract_ranking(self.perturbed_A, self.n_removed_link)

        if verbose:
            print(f'Missing Links: {self.links}')
            print(f'Predicted Links {ranks}')
        print(f"Structural Consistency: {self.evaluate(ranks)}")
