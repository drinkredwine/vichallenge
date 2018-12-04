import implicit
import numpy as np
from scipy.sparse import csr_matrix


def train_model(item_user_data) -> []:
    """"Returns trained model"""
    model = implicit.als.AlternatingLeastSquares(factors=50)

    model.fit(item_user_data)

    return model


def recommend(userid, model, user_items):
    """Returns recommended items for give userid"""
    return model.recommend(userid, user_items)


# def related(itemid):
#     """"Returns related items for given itemid"""
#     return model.similar_items(itemid)


def _create_sparse_matrix(ar):
    rows, r_pos = np.unique(ar[:, 0], return_inverse=True)
    cols, c_pos = np.unique(ar[:, 1], return_inverse=True)

    pivot_table = np.zeros((len(rows), len(cols)))
    pivot_table[r_pos, c_pos] = 1

    matrix = csr_matrix(pivot_table)

    return matrix


def _load_data(file) -> np.array:
    """Loads user, item, event, timestamp from CSV """
    return np.loadtxt(file, delimiter=',', skiprows=1, usecols=[0, 1])


def load_data(file) -> np.array:
    """Loads user, item, event, timestamp from CSV """
    ar = _load_data(file)
    matrix = _create_sparse_matrix(ar)
    return matrix
