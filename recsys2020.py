import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sys
sys.path.append("/Users/alessiorussointroito/Documents/GitHub/Structural-Perturbation-Method")

from SPM_fast import SPM

if __name__=="__main__":
    n_eigen = int(sys.argv[1])
    k = int(sys.argv[2])
    
    df = pd.read_csv(
        "/Users/alessiorussointroito/Desktop/Telegram Desktop/recommender-system-2020-challenge-polimi/data_train.csv")

    le = LabelEncoder()
    df['new_col'] = le.fit_transform(df.col)

    row_size = len(df.row.unique())
    col_size = len(le.classes_)
    N_nodes = row_size + col_size

    # Data Preprocessing
    """
    Siccome un tipico problema di recsys è rappresentabile con una rete bipartita, dove in un set ci sono gli user e 
    dall' altro gli items, in questa rete dobbiamo distinguere tutti i nodi. Di conseguenza supponiamo che gli items  
    rappresentino i nodi che vanno da 0 a 25974, gli user invece vanno da 24895 a 24896 + |users|
    """

    df['row'] = df.row + col_size

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(df.row, df.new_col, test_size=0.20, random_state=42)

    # Make the urm symmetric, creating 2 urm and summing them. The two matrices represent upper triangle and down triangle
    adj_down = sps.csr_matrix((np.ones(X_train.shape[0]), (X_train, y_train)), shape=(N_nodes, N_nodes))
    adj_up = sps.csr_matrix((np.ones(X_train.shape[0]), (y_train, X_train)), shape=(N_nodes, N_nodes))
    adj = adj_up + adj_down

    test_indices = X_test.unique()
    test_indices = np.sort(test_indices)
    test = pd.DataFrame({'row': X_test, 'target': y_test})

    targets = test.groupby(test.row)['target'].apply(lambda x: list(x))


    spm = SPM(adj, target=test_indices , p=0.2, n_eigen=n_eigen)
    rankings = spm.k_runs(k=k)


    # Evaluation

    def precision(is_relevant, relevant_items):
        # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
        return precision_score


    def recall(is_relevant, relevant_items):
        # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        recall_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
        return recall_score


    def MAP(is_relevant, relevant_items):
        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
        map_score = np.sum(p_at_k) / np.min([len(relevant_items), len(is_relevant)])
        return map_score


    n_users = len(test_indices)

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    num_eval = 0

    at = 10

    for i, user_id in enumerate(tqdm(test_indices)):
        relevant_items = targets[user_id]
        #recommended_items = rankings[user_id].argsort()[::-1][:at]
        recommended_items = rankings[i]

        # Filter Seen
        seen_indices = sps.find(adj[user_id])[1]    # Con [1] Prendiamo solo le colonne d'interesse della matrice di adiacenza
        mask = np.zeros(N_nodes, dtype=bool)
        mask[seen_indices] = True
        recommended_items[mask] = -np.inf

        # Recommend
        recommended_items = recommended_items.argsort()[::-1][:at]

        num_eval += 1

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        cumulative_precision += precision(is_relevant, relevant_items)
        cumulative_recall += recall(is_relevant, relevant_items)
        cumulative_MAP += MAP(is_relevant, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print(f"SPM n_eigen = {n_eigen} , k = {k}  \n Precision = {cumulative_precision} \n Recall = {cumulative_recall} \n MAP = {cumulative_MAP}")