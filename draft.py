import implicit
import numpy as np
import scipy.sparse as sp


def score(data, model, user_items, model_name='new_sub'):

    with open('data/sub_{}.csv'.format(model_name), 'w') as ff:
        ff.write("customer_id,product_id\n")
        for line in data:
            userid = np.int32(line)

            recs = model.recommend(userid, user_items=user_items)
            # print(userid, recs)
            for rec in recs:
                ff.write("{},{}\n".format(userid, rec[0]))

with open('data/vi_dataset_events.csv', 'r') as f:
    data = f.readlines()

d = np.zeros((86016 + 1, 28369 + 1), dtype=np.int32)

for line in data[1:]:
    user, item, event, timestamp = line.split(',')
    user = np.int32(user)
    item = np.int32(item)
    timestamp = np.int32(timestamp)

    if event == 'purchase':
        weight = 10
    elif event == 'add_to_cart':
        weight = 30
    else:
        weight = 1

    d[user, item] += weight

print(d)
mat = sp.csr_matrix(d.T)
print(mat)
model = implicit.als.AlternatingLeastSquares(factors=16)
model.fit(mat)
# print(model)

user_items = mat.T.tocsr()

# print(user)

with open('data/vi_challenge_uID.csv', 'r') as f:
    data = f.readlines()

score(data, model, mat.T, 'ALS')

model2 = implicit.bpr.BayesianPersonalizedRanking()
model2.fit(mat)

score(data, model2, mat.T, 'BPR')
