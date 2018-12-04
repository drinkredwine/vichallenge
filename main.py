# Recommender demo by drinkredwine 2018

import implicit
import numpy as np


def load_data() -> np.array:
    return np.load('data/vi_dataset_events')


def train_model(data) -> []:
    model = implicit.als.alternating_least_squares(factors=50)
    model.fit(item_user_data)


if __name__ == '__main__':
    item_user_data = load_data()
