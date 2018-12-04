import implicit
import numpy as np


def load_data() -> np.array:
    return np.load('data/vi_dataset_events')


def train_model(item_user_data: np.matrix) -> []:
    model = implicit.als.alternating_least_squares(factors=50)
    model.fit(item_user_data)
