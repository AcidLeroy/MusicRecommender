import h5py
import argparse
import numpy as np
import pandas as pd
from pyspark import SparkContext, SparkConf
import distance_metrics as dm


def get_features(file_name, features):
    f = h5py.File(file_name, 'r')
    data = f['/metadata/songs'][()]
    data = data[features]
    #my_nans = np.isnan(data[['artist_hotttnesss', 'artist_familiarity']])
    df = pd.DataFrame(data)
    df = df.fillna(0)
    return df.as_matrix()

def get_item_similarity(features):
    conf = SparkConf() \
        .setAppName("Collaborative Filter") \
        .set("spark.executor.memory", "6gb")
    sc = SparkContext(conf=conf)
    data = sc.parallelize(features[:,1:])
    print('Calculating the distance')
    data = dm.calculate_distance(data)
    data.saveAsTextFile('file:///Users/cody/saved_similarity.txt')




def main():
    parser = argparse.ArgumentParser(description='Create item similarity matrix.')
    parser.add_argument('hdf5_path', type=str, nargs=1, help='location of msd_summary_file.h5')
    parser.add_argument('--features', type=str, nargs='+', default=['song_id', 'artist_hotttnesss', 'artist_familiarity'],
                        help='list of features to extract from the hdf5 file')
    args = parser.parse_args()
    features = get_features(args.hdf5_path[0], args.features)
    print(features[:10])
    get_item_similarity(features)

if __name__ == '__main__':
    main()

