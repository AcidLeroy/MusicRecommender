from __future__ import print_function

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext, SparkConf
from src.parser import parse_line
from src.normalize import by_max_count
import os
import shutil
import argparse
import itertools


def main():

    parser = argparse.ArgumentParser(description='Create a collaborative filtering system for music ratings.')
    parser.add_argument('path', type=str, nargs=2,
                        help='collaborative_filter <absolute path to training file> <absolute path to testing file>')

    args = parser.parse_args()
    train_path = args.path[0]
    test_path = args.path[1]
    train_dataFile = 'file:///{}'.format(train_path)
    test_dataFile = 'file:///{}'.format(test_path)
    print("Train dataFile = ", train_dataFile)
    print("Test  dataFIle = ", test_dataFile)

    collaborative_filter(train_dataFile, test_dataFile)

def collaborative_filter(train_dataFile, test_dataFile):

    conf = SparkConf() \
        .setAppName("Collaborative Filter") \
        .set("spark.executor.memory", "5g")
    sc = SparkContext(conf=conf)


    # #             TRAINING            # #
    # Load and parse the data
    data = sc.textFile(train_dataFile)
    # #             Normalize start         # #
    print('Training normalization started')
    dataKV = data.map(lambda x: (x.split('\t')[0], x))
    userPlays = data.map(lambda x: (x.split('\t')[0], float(x.split('\t')[2])))
    userMax   = userPlays.foldByKey(0,max)
    userJoin = dataKV.join(userMax)
    Ndata = userJoin.map(lambda x: (x[0] + ' ' + x[1][0].split("\t")[1] + ' ' + str(5*float(x[1][0].split("\t")[2])/x[1][1])))
    print(' Training normalization ended')
    # #             Normalize end           # #
    ratings_map = Ndata.map(parse_line)
    num_ratings = ratings_map.count()
    num_users = ratings_map.map(lambda r: r['user']['hash']).distinct().count()
    num_songs = ratings_map.map(lambda r: r['song']['hash']).distinct().count()
    print("Got {} ratings, with {} distinct songs and {} distinct users".format(num_ratings,
                                                                                num_users,
                                                                                num_songs))
    ratings = ratings_map.map(lambda l: Rating(l['user']['hash'], l['song']['hash'], l['rating']))
    ratings_valid = ratings.sample(False,0.1,12345)
    ratings_train = ratings.subtract(ratings_valid)


    print(20*'-','TRAINING STARTED',20*'-')
    ranks = [8]
    lambdas = [1.0, 10.0, 5.0]
    numIters = [10]
    bestModel = None
    bestValidationMSE = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1
    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        print(ranks, lmbda, numIter)
        model = ALS.train(ratings_train, rank, numIter, lmbda)
        testdata = ratings_valid.map(lambda p: (p[0], p[1]))
        predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
        ratesAndPreds = ratings_valid.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
        MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        if (MSE < bestValidationMSE):
            bestModel = model
            bestValidationMSE = MSE
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter
    # evaluate the best model on the test set
    #model = ALS.train(ratings, rank, numIterations)
    print(20*'-','TRAINING FINISHED',20*'-')









    # #             TESTING             # #
    # # Evaluate the model on testing data
    print(20*'-','TESTING STARTED',20*'-')
    test_data = sc.textFile(test_dataFile)
    # #             Normalize start           # #
    print('testing normalization started')
    dataKV = test_data.map(lambda x: (x.split("\t")[0], x))
    userPlays = data.map(lambda x: (x.split("\t")[0], float(x.split("\t")[2])))
    userMax   = userPlays.foldByKey(0,max)
    userJoin = dataKV.join(userMax)
    Ndata = userJoin.map(lambda x: (x[0] + ' ' + x[1][0].split("\t")[1] + ' ' + str(5*float(x[1][0].split("\t")[2])/x[1][1])))
    print('testing normalization ended')
    # #             Normalize end           # #
    test_ratings_map = Ndata.map(parse_line)
    test_ratings = test_ratings_map.map(lambda l: Rating(l['user']['hash'], l['song']['hash'], l['rating']))
    testdata = test_ratings.map(lambda p: (p[0], p[1]))
    predictions = bestModel.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = test_ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    print("Mean Squared Error = " + str(MSE))
    print(20*'-','TESTING FINISHED',20*'-')


    # Save and load model
    path = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(path+'/myModelPath'):
        shutil.rmtree(path+'/myModelPath')
    path = 'file:///' + path + '/myModelPath'
    print('\n',20*'-','MODEL SAVED at',20*'-')
    print(path)
    print(50*'-')
    model.save(sc, path)
    sameModel = MatrixFactorizationModel.load(sc, path)


if __name__ == '__main__':
    main()
