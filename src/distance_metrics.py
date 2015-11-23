from pyspark import SparkContext, SparkConf
import numpy as np
import scipy.spatial.distance as ssd

def calculate_distance(rdd_in, distance_func):
    """
    Calculate any distance metric of an RDD
    :param rdd_in (RDD): Rows = sample of features from each item, Cols = features (NxD matrix)
    :param distance_func (function handle): This is any function from the scipy.spatial.distance library. the cosine and euclidean
    distances are both valid.
    :return: An RDD of (1xN^2) elements for distances between each sample to every other sample
    """
    rdd_out = rdd_in.cartesian(rdd_in).map(lambda x: distance_func(x[0], x[1]))
    return rdd_out


def test_calculate_distance():
    conf = SparkConf() \
        .setAppName("Collaborative Filter") \
        .set("spark.executor.memory", "100mb")
    sc = SparkContext(conf=conf)

    my_array = np.array([[1,2,3], [4,5,6], [7,8,9]])
    expected_value = np.zeros((3,3))

    # Calculate the truth
    for i_idx, i_val in enumerate(my_array):
        for j_idx, j_val in enumerate(my_array):
            expected_value[i_idx][j_idx] = ssd.cosine(i_val, j_val)

    rdd = sc.parallelize(my_array)
    dist = np.array(calculate_distance(rdd, ssd.cosine).collect())
    actual_value = dist.reshape((3,3))

    np.testing.assert_array_equal(expected_value, actual_value)


def main():
    test_calculate_distance()

if __name__ == '__main__':
    main()

