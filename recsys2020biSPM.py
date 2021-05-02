import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sys

sys.path.append("/Users/alessiorussointroito/Documents/GitHub/Structural-Perturbation-Method")

# from SPM_fast import SPM
from BiSPM import BiSPM

if __name__ == "__main__":
    n_svs = int(sys.argv[1])
    k = int(sys.argv[2])
    p = float(sys.argv[3])

    df = pd.read_csv(
        "/Users/alessiorussointroito/Downloads/Telegram Desktop/recommender-system-2020-challenge-polimi/data_train.csv")

    le = LabelEncoder()
    df['new_col'] = le.fit_transform(df.col)

    row_size = len(df.row.unique())
    col_size = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(df.row, df.new_col, test_size=0.20, random_state=3)

    bip_adj = sps.csr_matrix((np.ones(X_train.shape[0]), (X_train, y_train)), shape=(row_size, col_size))

    test_indices = X_test.unique()
    test_indices = np.sort(test_indices)
    test = pd.DataFrame({'row': X_test, 'target': y_test})

    targets = test.groupby(test.row)['target'].apply(lambda x: list(x))

    bspm = BiSPM(bip_adj, target=test_indices, n_sv=n_svs, p=p)
    rankings = bspm.k_runs(k=k)


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
        # recommended_items = rankings[user_id].argsort()[::-1][:at]
        recommended_items = rankings[i]

        # Filter Seen:
        # 1. Remove items already seen by the user
        seen_indices = sps.find(bip_adj[user_id])[1]  # Con [1] Prendiamo solo le colonne d'interesse della matrice di adiacenza
        mask = np.zeros(bip_adj.shape[1], dtype=bool)
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

    print(
        f"SPM n_eigen = {n_svs} , k = {k}  \n Precision = {cumulative_precision} \n Recall = {cumulative_recall} \n MAP = {cumulative_MAP}")
