import pandas as pd
import numpy as np
from content_filter import append_cluster_id, get_new_ratings

def main():
    stored_file = '/Users/cody/extracted_song_features.h5'
    training_file = '/Users/cody/Downloads/EvalDataYear1MSDWebsite/small_training_data.txt'
    testing_file = '/Users/cody/Downloads/EvalDataYear1MSDWebsite/testing_data.txt'
    song_features = pd.read_hdf(stored_file, '/features')
    training_data = pd.read_csv(training_file, sep='\t', names=['user_id', 'song_id', 'rating'])
    testing_data = pd.read_csv(testing_file, sep='\t', names=['user_id', 'song_id', 'rating'])
    testing_data = append_cluster_id(testing_data, song_features[['song_id', 'cluster']])
    training_data = append_cluster_id(training_data, song_features[['song_id', 'cluster']])
    new_ratings = get_new_ratings(testing_data, training_data)
    new_ratings = new_ratings.dropna()
    new_ratings[['user_id', 'song_id', 'rating']].to_csv('augmented_test_data.txt', sep='\t', header=False, index=False)

if __name__ == '__main__':
    main()
