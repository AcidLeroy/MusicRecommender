from __future__ import print_function
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal

def append_cluster_id(rating_data, features):
    """
    This function is designed to append a new column at the end of the rating data to identify what cluster
    a particular song belongs to.
    :param rating_data:
    :param features:
    :return:
    """
    rating_data = rating_data.merge(features, on='song_id', how='left')
    return rating_data

def get_new_ratings(test_data, train_data):
    """
    Given the training and testing data, both with the cluster id appended to them, fill in the missing ratings
    in the testing data. If there are any ratings with the rating NaN, those rows should be dropped.
    :param test_data:
    :param train_data:
    :return:
    """

    user_group = train_data.groupby('user_id')

    def func(x):
        user = x['user_id']
        song_cluster_id = x['cluster']
        song_id = x['song_id']
        if user not in user_group.groups:
            return x

        user_ratings = user_group.get_group(user)
        song_ratings = user_ratings.groupby('song_id')
        if song_id in song_ratings.groups:
            x['rating'] = song_ratings.get_group(song_id)['rating'].iloc[0]
            return x

        cluster_ratings = user_ratings.groupby('cluster')
        if song_cluster_id in cluster_ratings.groups:
            x['rating'] = cluster_ratings.get_group(song_cluster_id)['rating'].mean()

        return x

    actual_value = test_data.apply(func, axis=1)
    return actual_value


def setup_test_data():
    training_data = pd.DataFrame({'user_id':['u1', 'u1', 'u2', 'u2'], 'song_id':['s1', 's2', 's3', 's4'],
                                  'rating':[5, 8, 10, 2]})
    testing_data = pd.DataFrame({'user_id':['u1', 'u1', 'u2', 'u2', 'u1'], 'song_id':['s5', 's4', 's1', 's2', 's1'],
                                  'rating':[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]})

    features = pd.DataFrame({'song_id':['s1', 's2', 's3', 's4', 's5'], 'cluster':[0, 0, 1, 2, 0]})

    return (training_data, testing_data, features)

def test_get_same_ratings():
    (train_data, test_data, features) = setup_test_data()
    train_data = append_cluster_id(train_data, features)
    new_train = train_data.copy()
    actual_value = get_new_ratings(train_data, new_train)
    print(actual_value)
    assert_frame_equal(train_data, actual_value)

def test_unseen_user():
    (train_data, test_data, features) = setup_test_data()
    test_data = append_cluster_id(test_data, features)
    train_data = append_cluster_id(train_data, features)
    unseen_user = pd.DataFrame({'song_id': ['s4'],
                                'rating':[np.NaN],
                                'user_id':['u99'],
                                'cluster':[0]})
    test_data = pd.concat([test_data, unseen_user], ignore_index=True)
    expected_value = test_data.copy()
    actual_value = get_new_ratings(test_data, train_data)
    print(actual_value)
    expected_value['rating'].iloc[0] = 6.5
    expected_value['rating'].iloc[4] = 5.0
    print(expected_value)
    assert_frame_equal(expected_value, actual_value)



def test_get_new_ratings():
    (train_data, test_data, features) = setup_test_data()
    test_data = append_cluster_id(test_data, features)
    train_data = append_cluster_id(train_data, features)

    actual_value = get_new_ratings(test_data, train_data)

    expected_value = test_data.copy()
    expected_value['rating'] = [6.5, np.NaN, np.NaN, np.NaN, 5.0]

    print('actual_value....')
    print(actual_value)
    print('expected_value....')
    print(expected_value)
    assert_frame_equal(expected_value, actual_value)



def test_append_cluster_id():
    (train_data, test_data, features) = setup_test_data()
    expected_cluster = test_data.copy()
    expected_cluster['cluster'] = [0, 2, 0, 0, 0]
    expected_cluster = expected_cluster.sort('song_id')

    actual_cluster = test_data.copy()
    actual_cluster = append_cluster_id(actual_cluster, features)
    actual_cluster = actual_cluster.sort('song_id')

    print(actual_cluster)
    print(expected_cluster)

    assert_frame_equal(expected_cluster, actual_cluster)


def main():
    test_append_cluster_id()
    test_get_new_ratings()
    test_get_same_ratings()
    test_unseen_user()

if __name__ == '__main__':
    main()