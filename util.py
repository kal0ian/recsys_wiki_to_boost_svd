import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from surprise import Dataset

USERS_COUNT = 943
ITEMS_COUNT = 1682
THRESHOLD = 0.05


def load_data(data_removed=0.95, test_size=0.20):
    all_data = load_movies_data()
    train_set, _ = train_test_split(all_data, test_size=data_removed)
    sparcity = 1 - (len(train_set) / (USERS_COUNT * ITEMS_COUNT))
    print("Sparcitiy: %.4f" % sparcity)
    return train_test_split(train_set, test_size=.20)


def load_movies_data():
    data = pd.DataFrame(Dataset.load_builtin("ml-100k").raw_ratings)
    data[0] = pd.to_numeric(data[0]) - 1
    data[1] = pd.to_numeric(data[1]) - 1
    del data[3]
    return data.values()


def load_similarities():
    similarities = pd.read_csv("item_similarities.csv")
    similarities['0'] = similarities['0']
    similarities['1'] = similarities['1']

    similarities_arr = np.zeros((ITEMS_COUNT, ITEMS_COUNT))
    for _, row in similarities.iterrows():
        if int(row['0']) != int(row['1']):
            similarities_arr[int(row['0']), int(row['1'])] = row['2']

    return similarities_arr


def generate_artificial_ratings(train_set, similarities):
    artificial = np.zeros((USERS_COUNT, ITEMS_COUNT))
    for item in range(ITEMS_COUNT):
        for user in range(USERS_COUNT):
            rating = 0
            all_user_ratings = train_set[train_set[:, 0] == user]
            sum_sim = 0
            for u, i, r in all_user_ratings:
                sim1 = similarities[item, int(i)]
                if (sim1 < THRESHOLD):
                    continue
                rating = rating + r * sim1
                sum_sim = sum_sim + sim1
            if sum_sim == 0:
                artificial[user, item] = -1
            else:
                artificial[user, item] = rating / sum_sim

    artificial_data = []
    for item in range(ITEMS_COUNT):
        for user in range(USERS_COUNT):
            if (artificial[user, item] != -1):
                artificial_data.append([user, item, artificial[user, item]])
    return np.array(artificial_data)


def stack_real_and_artificial_data(real_data, artificial_data):
    artificial_data = np.column_stack((artificial_data,
                                       np.zeros(artificial_data.shape[0])))
    real_data = np.column_stack((real_data, np.ones(real_data.shape[0])))
    return np.random.shuffle(np.vstack((artificial_data, real_data)))
