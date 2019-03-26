# -*- coding: utf-8 -*-
"""
@created on: 2019-03-22,
@author: Vivek A Gupta,

Description:

..todo::
"""

import pandas as pd
import numpy as np
import csv

movie_path = "ml-1m/movies.dat"
users_path = "ml-1m/users.dat"
ratings_path = "ml-1m/ratings.dat"

train_path = "ml-100k/u1.base"
test_path = "ml-100k/u1.test"

# movie_df = pd.read_csv(movie_path, sep='::', encoding='latin-1', header=None, engine='python')
# users_df = pd.read_csv(movie_path, sep='::', encoding='latin-1', header=None, engine='python')
# ratings_df = pd.read_csv(ratings_path, sep='::', encoding='latin-1', header=None, engine='python')


def get_data(train_path, test_path):
    training_set = []
    with open(train_path, 'r') as fp:
        for line in csv.reader(fp, dialect="excel-tab"):
            line = [int(i) for i in line]
            training_set.append(line)

    test_set = []
    with open(test_path, 'r') as fp:
        for line in csv.reader(fp, dialect="excel-tab"):
            line = [int(i) for i in line]
            test_set.append(line)

    training_set = np.array(training_set)
    test_set = np.array(test_set)

    nb_users = max(max(training_set[:,0]), max(test_set[:,0]))
    nb_movies = max(max(training_set[:,1]), max(test_set[:,1]))

    def convert(data):
        new_data = []
        for id_users in range(1,nb_users + 1):
            id_movies = data[:, 1][data[:, 0] == id_users]
            id_ratings = data[:, 2][data[:, 0] == id_users]
            ratings = np.zeros((nb_movies))
            ratings[id_movies - 1] = id_ratings
            new_data.append(list(ratings))
        return np.asarray(new_data)

    training_set = convert(training_set)
    test_set = convert(test_set)

    training_set[training_set == 0] = -1
    training_set[training_set == 1] = 0
    training_set[training_set == 2] = 0
    training_set[training_set >= 3] = 1

    test_set[test_set == 0] = -1
    test_set[test_set == 1] = 0
    test_set[test_set == 2] = 0
    test_set[test_set >= 3] = 1

    return training_set, test_set


def get_inference_data(test_data, train_data):
    new_train = []
    new_test = []
    for test, train in zip(test_data, train_data):
        if len(test[test >= 0]) > 0:
            new_train.append(train)
            new_test.append(test)
    return np.asarray(new_train), np.asarray(new_test)


# Divide into batches.
def divide_batches(input_batch, batch_size):
    output_batch = []
    for i in range(0, len(input_batch), batch_size):
        output_batch.append(input_batch[i: i + batch_size])
    return output_batch
