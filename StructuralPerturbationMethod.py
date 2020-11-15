import numpy as np
import math
from tqdm import tqdm, trange


# TODO Farlo tutto con scipy.sparse

class SPM(object):

    """
    Given a graph G = (V, E), we can construct the adjacency matrix A of the E links.
    We randomly select a fraction p of the links to constitute a perturbation set ΔE, while the rest of the links E − ΔE
    constitute the set E_r . Denote by A and ΔA the corresponding adjacency matrices; obviously, A = AR + ΔA.

    RecSys:
        - URM: user-item matrix
        - Dalla URM si crea una matrice quadrata item-item o user-user che corrisponde alla matrice
            di partenza del SPM
        - Si usa SPM e si ottiene una matrice quadrata A_signed item-item o user-user che corrisponde ad una similarità [come cosine]
        - Per tirare fuori gli scores si fa:
            1. Se la matrice quadrata iniziale era item-item: recommendations = URM * A_signed
            2. Se la matrice quadrata iniziale era user-user: recommendations = A_signed * URM

    """

    def __init__(self, A, p=None, delta_A=None, n_remove_link=None):
        """
        :param A: original adjacency matrix that would be decomposed in A = A_r + delta_A
        :param p: fraction of links to generate a perturbation set delta_E
        :param delta_A: particular perturbation. If None, a random perturbation is computed
        :return:
        """

        assert (p is not None and delta_A is None and n_remove_link is None) or \
               (p is None and delta_A is not None and n_remove_link is not None) , "Wrong input choice!"

        self.A = A
        self.p = p
        self.N = self.A.shape[0]
        self.delta_A = delta_A
        self.n_removed_link = n_remove_link

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
        return matrix

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


    def check_degenerate_eig(self, lambda_k, x_k):
        """Procedure to check and correct degenerate eigenvalue
        Degenerate eigenvalues means that M eigenvectors correspond to the same eigenvalue, or, on the other hand,
        some eigenvalues have the same value. In order to consider a perturbation, we need to transform these degenerate
        eigenvalues in a non-degenerate eigenvalues"""


        from collections import defaultdict, OrderedDict

        eigen_dict = defaultdict(list)
        tmp_list = list(zip(lambda_k, range(len(lambda_k))))

        # Create a dict with lambda_k --> [List of indices of eigenvector associated to the same eigenvalue lambda_k]
        for key, value in tmp_list:
            eigen_dict[key].append(value)

        # Final Dict to restore the order of the lambda_k:
        # key= index of the eigenvector  --> value: (lambda_k, delta_lambda_k, x_k) or (lambda_k, d_sign_lambda_k, x_sign_k)

        final_delta_lambda_k = OrderedDict()
        final_x_k = OrderedDict()

        # Correction Procedure
        for key, value in eigen_dict:
            if len(value) > 1:
                print(f"Correction for eigenvalue {key}")
                m = np.array([x_k[:, i] for i in value]).T

                # Transform eigenvectors (they are all linearly independent) associated to the same eigenvalue in a basis
                # via Gram-Schmidt algorithm
                q,r = np.linalg.qr(m)

                # Create W : see "Support Information 'Case of degenerate eigenvalues'
                # https://www.researchgate.net/publication/272372246_Toward_link_predictability_of_complex_networks "

                W = q.T.dot(self.delta_A).dot(q)    # --> W is an MxM matrix

                # Following the paper, we want to find eigenvalues (delta_lambda_k)/eigenvectors(B_k)
                # for W*B_k = delta_lambda_k * B_k .
                # Delta_sign_lambda_k represents the corrected non-degenerate eigenvalue
                # x_sign_k represent the unique eigenvector associated to the corrected eigenvalue

                d_sign_lambda_k, B_k = np.linalg.eig(W)
                x_sign_k = q.dot(B_k.T)   # q: NxM   B_k: MxM


                for i in range(len(value)):
                    #res_dict[value[i]] = (d_sign_lambda_k[i], x_sign_k[:, i])

                    final_delta_lambda_k[value[i]] = d_sign_lambda_k[i]
                    final_x_k[value[i]] = x_sign_k[:, i]

            elif len(value) == 1:
                # compute delta_lambda_k for normal eigenvalues
                num = x_k[:, value[0]].T.dot(self.delta_A).dot(x_k[:, value[0]])
                den = x_k[:, value[0]].T.dot(x_k[:, value[0]])
                #res_dict[value[0]] = (num / den,  x_k[:, value[0]])

                final_delta_lambda_k[value[0]] = num / den
                final_x_k[value[0]] = x_k[:, value[0]]

            else:
                raise Exception(f"Error: eigenvalue {key} has no index in its list {value}")

        # Return delta_lambda_k and x_k as matrix
        res_delta_lambda_k = np.array(list(final_delta_lambda_k.values()))
        res_x_k = np.array(list(final_x_k)).T

        return res_delta_lambda_k, res_x_k

    def compute_perturbed_matrix(self, A_r, delta_A):

        lambda_k, x_k = np.linalg.eig(A_r)

        if len(set(lambda_k)) != len(lambda_k):

            print("Degenerate-Eigenvalues! --> Correction..")
            delta_lambda_k, x_k = self.check_degenerate_eig(lambda_k, x_k)   #TODO controllare che tutto funzioni

        else:
            print("Non-degenerate Eigenvalues. Compute delta_lambda_k")
            delta_lambda_k = self.compute_delta_lambda(x_k, delta_A)

        perturbed_A = np.zeros(A_r.shape)

        # perturbed_A = perturbed_A + ((lambda_k + delta_lambda_k)*(x_k.dot(x_k.T)))
        for k in range(A_r.shape[0]):
            #perturbed_A += (lambda_k[k] + delta_lambda_k[k]) * np.dot(x_k[:, k][:, None], x_k[:, k][None, :])
            perturbed_A += (lambda_k[k] + delta_lambda_k[k]) * np.dot(x_k[:, k][:, None], x_k[:, k][None, :])           # TODO prodotto tra matrici?


        # perturbed_A Cleaning: check the existing link in A_r [i.e. in E_r] and set those positions = -np.inf in
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

        print("Get ranking..")
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
        self.split_init(isSingle=True)

        self.perturbed_A = self.compute_perturbed_matrix(self.A_r, self.delta_A)
        ranks = self.extract_ranking(self.perturbed_A, self.n_removed_link)

        if verbose:
            print(f'Missing Links: {self.links}')
            print(f'Predicted Links {ranks}')
        print(f"Structural Consistency: {self.evaluate(ranks)}")

        return self.links, ranks

    def k_runs(self, k, verbose=True):
        assert (k != 1), "k must be != 1, call run otherwise"

        res = np.zeros((self.N, self.N))
        for i in trange(k, desc='Iteration'):
            self.split_init(isSingle=False)

            res += self.compute_perturbed_matrix(self.A_r, self.delta_A)

        res /= k

        ranks = self.extract_ranking(res, self.n_removed_link)

        if verbose:
            print(f'Missing Links: {self.links}')
            print(f'Predicted Links {ranks}')
        print(f"Structural Consistency: {self.evaluate(ranks)}")
        return self.links, ranks
