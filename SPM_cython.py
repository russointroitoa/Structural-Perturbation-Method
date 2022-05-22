import numpy as np
import math
from tqdm import tqdm, trange
import scipy.sparse as sps

import sys
sys.path.append("/Users/alessiorussointroito/Documents/GitHub/Structural-Perturbation-Method/")

from cython_files.compute_perturbed_A import compute_perturbed_A


class SPM(object):

    def __init__(self, A, target, p=None, delta_A=None, n_remove_link=None, n_eigen=300):
        """
        :param A: original adjacency matrix that would be decomposed in A = A_r + delta_A.   NEED TO BE DENSE
        :param p: fraction of links to generate a perturbation set delta_E
        :param delta_A: particular perturbation. If None, a random perturbation is computed
        :param k: Number of eigevalues (and eigenvectors) to extract. Max=A.shape[0]-1 but it will slow down a lot
        :return:
        """

        assert (p is not None and delta_A is None and n_remove_link is None) or \
               (p is None and delta_A is not None and n_remove_link is not None) , "Wrong input choice!"

        assert (sps.isspmatrix_csr(A) or sps.isspmatrix_csc(A)), "A is not csr matrix"

        self.A = A
        self.p = p
        self.N = self.A.shape[0]
        #self.delta_A = delta_A
        self.n_removed_link = n_remove_link
        self.n_eigen = n_eigen
        self.target = np.sort(target)


    def split_init(self, isSingle=True):
        if self.delta_A is not None:
            # self.n_removed_link = n_remove_link
            # self.delta_A = delta_A
            self.A_r = self.A - self.delta_A
            self.links = np.where(np.triu(self.delta_A) == 1)
            self.links = list(zip(self.links[0], self.links[1]))
        else:
            if isSingle:
                np.random.seed(42)

            #self.n_removed_link = math.ceil(self.N * self.p)    # TODO: percentage on number of links, not nodes

            self.n_removed_link = math.ceil((self.A.nnz/2) * self.p)

            # Extract a fraction p of links from A to generate delta_A
            links = sps.find(sps.triu(self.A))[:2]  # Tuple (x, y) with x and y are lists of indices, we consider only
                                                        # the upper triangle of the matrix and reflect the symmetry of the
                                                        # matrix in self.create_adj_matrix()

            indices = np.random.randint(0, links[0].shape[0], self.n_removed_link)

            r = links[0][indices]
            c = links[1][indices]

            self.links = list(zip(r,c))

            # Create delta_A
            rows = np.append(r,c)
            cols = np.append(c,r)
            self.delta_A = self.create_adj_matrix(self.N, rows, cols)

            # Create A_r = A - delta_A
            self.A_r = self.A - self.delta_A


    def create_adj_matrix(self, N, rows, cols):
        data = np.ones(rows.shape[0])
        matrix = sps.csr_matrix((data, (rows, cols)), shape=(N, N))
        return matrix

    def compute_delta_lambda(self, v, delta_A):
        """
        v : eigenvectors
        delta_A : perturbation matrix
        """
        res = np.empty(self.n_eigen)

        for k in range(self.n_eigen):
            # num = v[:, k].T.dot(delta_A).dot(v[:, k])
            # Change delta_A with xT in order to use sparse dot: sparse matrix need to be before
            num = (delta_A.T.dot(v[:, k])).T.dot(v[:, k])
            den = v[:, k].T.dot(v[:, k])
            res[k] = num / den

        return res

    def compute_perturbed_matrix(self, A_r, delta_A):
        self.pbar.set_description("Computing Eigen")
        lambda_k, x_k = sps.linalg.eigsh(A_r, k=self.n_eigen)  # Eigenvalues/eigenvector extraction for symmetric matrix

        # Normalize lambda_k by its norm
        lambda_k /= np.linalg.norm(lambda_k)

        if len(set(lambda_k)) != len(lambda_k):
            print("Degenerate-Eigenvalues! --> Correction..")
            delta_lambda_k = None
            exit()

        else:
            #print("Non-degenerate Eigenvalues. Compute delta_lambda_k")
            self.pbar.set_description("Computing delta lambdas")
            delta_lambda_k = self.compute_delta_lambda(x_k, delta_A)

        # Compute Perturbed Matrix
        # Perturbed_A : Non-squared matrix. Restrict computation to only relevant items
        x_k = x_k.astype(np.float32)
        lambda_k = lambda_k.astype(np.float32)
        delta_lambda_k = delta_lambda_k.astype(np.float32)

        self.pbar.set_description("Computing Perturbed matrix")
        # perturbed_A = perturbed_A + ((lambda_k + delta_lambda_k)*(x_k.dot(x_k.T)))
        perturbed_A = compute_perturbed_A(self.n_eigen, x_k, lambda_k, delta_lambda_k, self.target, self.N)


        # perturbed_A Cleaning: check the existing link in A_r [i.e. in E_r] and set those positions = -np.inf in
        # perturbed_A such that these links don't occur in the ranking to extract the most probable missing link.
        # What we are doing is to consider the set U - E_r, where U is the universe of possible links given N nodes
        #perturbed_A[np.where(A_r == 1)] = -np.inf
        self.pbar.set_description("Remove seen ")

        # perturbed_A[sps.find(A_r)[:2]] = -np.inf  #  Filter seen sul main

        # Moreover we set to -np.inf also the element (i,i) in the diagonal since we do not have self-links
        # np.fill_diagonal(perturbed_A, -np.inf)

        # Set the [i, i] of the matrix to -np.inf to avoid self-links
        for i, e in enumerate(self.target):
            perturbed_A[i, e] = -np.inf

        return perturbed_A

    def extract_ranking(self, perturbed_A, at):
        # ranking = perturbed_A.ravel().argsort()[::-1]

        # Since the matrix is symmetric, we can consider the upper triangle and consider the first half of results
        # (the second half represents the element of the low triangle of the matrix, set to zero)

        #print("Get ranking..")
        up_matrix_ranking = np.triu(perturbed_A).ravel().argsort()[::-1]
        ranking = up_matrix_ranking[:len(up_matrix_ranking) // 2]
        ranking = [(int(x / self.N), x % self.N) for x in ranking][:at]
        return ranking

    def evaluate(self, predicted_links):
        # We need to replicate each link such that there are both (x,y) and (y, x)
        preds = set(predicted_links).union(set([(y, x) for (x, y) in predicted_links]))
        links = set(self.links).union(set([(y, x) for (x, y) in self.links]))

        n_missing_links = len(links)
        correct_predicted = len(set(preds).intersection(links))
        return correct_predicted / n_missing_links

    def run(self, verbose=True):
        self.split_init(isSingle=True)

        self.perturbed_A = self.compute_perturbed_matrix(self.A_r, self.delta_A)
        ranks = self.extract_ranking(self.perturbed_A, self.n_removed_link)

        if verbose:
            print(f'Missing Links: {self.links}')
            print(f'Predicted Links {ranks}')
        print(f"Structural Consistency: {self.evaluate(ranks)}")

        return self.links, ranks

    def k_runs(self, k, verbose=True, save=False):
        assert (k != 1), "k must be != 1, call run otherwise"

        # Progress bar
        self.pbar = tqdm(range(k))

        #res = np.zeros((self.N, self.N))
        res = np.zeros((len(self.target), self.N))
        #for i in trange(k, desc='Iteration'):
        for i in self.pbar:
            self.pbar.set_description("Split")
            self.split_init(isSingle=False)

            res += self.compute_perturbed_matrix(self.A_r, self.delta_A)

            # Reset delta_A
            self.delta_A = None
        res /= k

        if save:
            np.save(f"SPM_similarity_{self.n_eigen}_{k}.npy", res)

        """
        # TODO non ha senso visto che self.links cambia da iterazione ad iterazione
        self.pbar.set_description("Extract rankings")
        ranks = self.extract_ranking(res, self.n_removed_link)

        if verbose:
            print(f'Missing Links: {self.links}')
            print(f'Predicted Links {ranks}')
        print(f"Structural Consistency: {self.evaluate(ranks)}")
        return self.links, ranks
        """

        return res
