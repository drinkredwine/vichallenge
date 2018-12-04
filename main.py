# Recommender demo by drinkredwine 2018

import app.models as m
import numpy as np

if __name__ == '__main__':
    item_user_data = m.load_data('data/vi_dataset_events.csv')
    model = m.train_model(item_user_data)

with open('data/vi_challenge_uID.csv', 'r') as f:
    data = f.readlines()

for line in data:
    userid = np.float(line)

    recs = m.recommend(userid, d)
    print(userid, recs)



