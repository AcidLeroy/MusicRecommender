from src.parser import parse_line

def by_max_count(rdd_in):
    """
    Assuming that rdd_in has is a triplet of (user_id, song_id, and
    play_count (int))
    :param rdd_in: RDD that represents a list of triplets.
    :return: Returns an RDD with with play count normalized
    """
    dataKV = rdd_in.map(lambda x: (x[0], x))
    userPlays = dataKV.map(lambda x: (x[0], x[1][2]))
    userMax = userPlays.foldByKey(0,max)
    userJoin = dataKV.join(userMax)
    Ndata = userJoin.map(lambda x: (x[0], x[1][0][0], 10*x[1][0][2]/x[1][1]))
    return Ndata

def format_triplets(rdd_in):
    ratings_dict = rdd_in.map(parse_line)
    ratings_triplet = ratings_dict.map(lambda x: (x['user']['hash'], x['song']['hash'], x['rating']))
    return (ratings_dict, ratings_triplet)