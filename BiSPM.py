import numpy as np
import math
from tqdm import tqdm, trange
import scipy.sparse as sps
from sklearn.utils.extmath import randomized_svd

from cython_files.compute_perturbed_A import compute_perturbed_B

np.random.seed(42)

class BiSPM(object):


    def __init__(self, bipartite_adj, target, n_sv, p=0.1):
        """

        :param bipartite_adj:
        :param target:
        :param p:
        """

        self.B = bipartite_adj
        self.target_users = np.sort(target)
        self.p = p
        self.shape = self.B.shape
        self.n_sing_values = n_sv
        self.seed_list = None

    def create_adj_matrix(self, shape, rows, cols):
        """

        :param shape: tuple(N,M)
        :param rows: rows indices of nnz values
        :param cols: cols indices of nnz values
        :return:
        """
        data = np.ones(rows.shape[0])
        matrix = sps.csr_matrix((data, (rows, cols)), shape=shape)
        return matrix


    def split_init(self,):
    
        self.n_removed_link = math.ceil((self.B.nnz) * self.p)

        # Extract a fraction p of links from A to generate delta_A
        links = sps.find(sps.triu(self.B))[:2]  # Tuple (x, y) with x and y are lists of indices, we consider only
                                                    # the upper triangle of the matrix and reflect the symmetry of the
                                                    # matrix in self.create_adj_matrix()

        indices = np.random.randint(0, links[0].shape[0], self.n_removed_link)

        r = links[0][indices]
        c = links[1][indices]

        #self.links = list(zip(r,c)) # TODO non credo serva
        self.delta_B = self.create_adj_matrix(self.shape, r, c)

        # Create A_r = A - delta_A
        self.B_r = self.B - self.delta_B

    def compute_delta_sv(self, B_r, delta_B, sv, vt):
        delta = np.empty(self.n_sing_values)

        squared_1 = B_r.T.dot(delta_B)
        squared_2 = delta_B.T.dot(B_r)
        squared = squared_1 + squared_2

        for i in range(self.n_sing_values):
            delta[i] = (squared.T.dot(vt[i][:,None])).T.dot(vt[i][:,None])[0][0]
            delta[i] = delta[i] / (2*sv[i]*vt[i].dot(vt[i]))

        return delta

    def compute_perturbed_matrix(self, B_r, delta_B):
        self.pbar.set_description("SVD")
        # SVD
        #u, s, vt = sps.linalg.svds(B_r, k=self.n_sing_values)
        u, s, vt = randomized_svd(B_r, n_components=self.n_sing_values)

        self.pbar.set_description("Delta_svs")
        # Compute delta singular values
        delta_svs = self.compute_delta_sv(B_r, delta_B, s, vt)

        # Type conversion
        u = u.astype(np.float32)
        s = s.astype(np.float32)
        vt = vt.astype(np.float32)
        delta_svs = delta_svs.astype(np.float32)

        self.pbar.set_description("Computing Perturbed B")
        perturbed_B = compute_perturbed_B(self.n_sing_values, u, s, vt, delta_svs, self.target_users)

        return perturbed_B

    def k_runs(self, k, save=False):
        self.seed_list = np.random.randint(1024, size=k)

        self.pbar = tqdm(range(k))
        res = np.zeros((len(self.target_users), self.shape[1]))

        for i in self.pbar:
            seed = self.seed_list[i]
            np.random.seed(seed)
            self.pbar.set_description("Split")
            self.split_init()

            res += self.compute_perturbed_matrix(self.B_r, self.delta_B)

            # Reset delta_B
            self.delta_B = None
            self.B_r = None

        res /= k

        if save:
            pass

        return res

    def get_B_r(self):
        return self.B_r

    def get_delta_B(self):
        return self.delta_B

