from pyspark import SparkContext, SparkConf

def by_max_count(rdd_in):
    """
    Assuming that rdd_in has is a triplet of (user_id, song_id, and
    play_count (int))
    :param rdd_in: RDD that represents a list of triplets.
    :return: Returns an RDD with with play count normalized
    """
    dataKV = rdd_in.map(lambda x: (x.split(" ")[0], x))
    userPlays = rdd_in.map(lambda x: (x.split(" ")[0], float(x.split(" ")[2])))
    userMax   = userPlays.foldByKey(0,max)
    userJoin = dataKV.join(userMax)
    Ndata = userJoin.map(lambda x: (x[0], x[1][0].split(" ")[1], 5*float(x[1][0].split(" ")[2])/x[1][1]))
    return Ndata
