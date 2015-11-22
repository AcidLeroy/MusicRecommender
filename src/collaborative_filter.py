from __future__ import print_function

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext, SparkConf
from src.parser import parse_line
from src.normalize import by_max_count, format_triplets
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

    train_ratings = get_ratings(sc, train_dataFile)

    ratings_valid = train_ratings.sample(False, 0.1, 12345)
    ratings_train = train_ratings.subtract(ratings_valid)


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
        print(rank, lmbda, numIter)
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
    test_ratings = get_ratings(sc, test_dataFile)

    testdata = test_ratings.map(lambda p: (p[0], p[1]))
    predictions = bestModel.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = test_ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    MAE = ratesAndPreds.map(lambda r: (abs(abs(r[1][0]) - abs(r[1][1])))).mean()

    print("Mean Squared Error = " + str(MSE))
    print("Mean Absolute Error = " + str(MAE))
    print("Root Mean Square Error = ", str(MSE**.5))
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


def get_ratings(sc, data_file):

    data = sc.textFile(data_file)
    # #             Normalize start         # #
    print('Training normalization started')
    data_dict, data_triplet = format_triplets(data)
    data_triplet = by_max_count(data_triplet)
    print(' Training normalization ended')
    # #             Normalize end           # #
    num_ratings = data_triplet.count()
    num_users = data_triplet.map(lambda r: r[0]).distinct().count()
    num_songs = data_triplet.map(lambda r: r[1]).distinct().count()
    print(100 * '//')
    print("Got {} ratings, with {} distinct songs and {} distinct users".format(num_ratings,
                                                                                num_users,
                                                                                num_songs))
    print(100 * '//')
    train_ratings = data_triplet.map(lambda l: Rating(l[0], l[1], l[2]))
    return train_ratings


if __name__ == '__main__':
    main()
