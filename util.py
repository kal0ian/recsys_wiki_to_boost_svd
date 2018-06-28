import pandas as pd
import numpy as np
from datetime import timedelta
import time
from itertools import product

from sklearn.model_selection import train_test_split

from surprise import Dataset

USERS_COUNT = 943
ITEMS_COUNT = 1682
THRESHOLD = 0.1


def load_data(data_removed=0.95, test_size=0.20):
    print("Loading data...")
    all_data = load_movies_data()
    train_set, _ = train_test_split(all_data, test_size=data_removed)
    sparcity = 1 - (len(train_set) / (USERS_COUNT * ITEMS_COUNT))
    print("Loading data completed. Sparcitiy: %.4f" % sparcity)
    train_set, test_set = train_test_split(train_set, test_size=.20)
    train_set = np.column_stack((train_set, np.ones(train_set.shape[0])))
    return train_set, test_set, sparcity


def load_movies_data():
    data = pd.DataFrame(Dataset.load_builtin("ml-100k").raw_ratings)
    data[0] = pd.to_numeric(data[0]) - 1
    data[1] = pd.to_numeric(data[1]) - 1
    del data[3]
    return data.values


def load_similarities():
    print("Loading similarities...")
    start_time = time.time()
    similarities = pd.read_csv("item_similarities.csv")
    similarities['0'] = similarities['0']
    similarities['1'] = similarities['1']

    similarities_arr = np.zeros((ITEMS_COUNT, ITEMS_COUNT))
    for _, row in similarities.iterrows():
        if int(row['0']) != int(row['1']):
            similarities_arr[int(row['0']), int(row['1'])] = row['2']

    elapsed = time.time() - start_time
    print("Loading similarities completed in: ",
          str(timedelta(seconds=elapsed)))
    return similarities_arr


def generate_artificial_ratings(train_set, test_set, similarities):
    print("Loading artificial ratings...")
    start_time = time.time()
    artificial = np.zeros((USERS_COUNT, ITEMS_COUNT))
    for item, user in product(range(ITEMS_COUNT), range(USERS_COUNT)):
        rating = 0
        all_user_ratings = train_set[train_set[:, 0] == user]
        sum_sim = 0
        for u, i, r, label in all_user_ratings:
            sim1 = similarities[item, int(i)]
            if (sim1 < THRESHOLD):
                continue
            rating = rating + r * sim1
            sum_sim = sum_sim + sim1
        if sum_sim == 0:
            artificial[user, item] = -1
        else:
            artificial[user, item] = rating / sum_sim

    ratings = {}
    for u, i, _, _ in train_set:
        if u not in ratings:
            ratings[u] = []
        ratings[u].append(i)

    for u, i, _ in test_set:
        if u not in ratings:
            ratings[u] = []
        ratings[u].append(i)

    artificial_data = []

    for item, user in product(range(ITEMS_COUNT), range(USERS_COUNT)):
        if (artificial[user, item] != -1):
            if user not in ratings or item not in ratings[user]:
                artificial_data.append([user, item, artificial[user, item]])

    elapsed = time.time() - start_time
    print("Loading artificial ratings completed in: ",
          str(timedelta(seconds=elapsed)))
    artificial_data = np.array(artificial_data)
    return np.column_stack((artificial_data,
                            np.zeros(artificial_data.shape[0])))
