from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext, SparkConf
import os

def parse_userid()

spark_home = os.environ['SPARK_HOME']

conf = SparkConf() \
      .setAppName("Collaborative Filter") \
      .set("spark.executor.memory", "5g")

sc = SparkContext(conf=conf)


dataFile = 'file:////Users/cody/Downloads/EvalDataYear1MSDWebsite/year1_test_triplets_hidden.txt'

# Load and parse the data
data = sc.textFile(dataFile.format(spark_home))

ratings = data.map(lambda l: l.split()).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

# Save and load model
path = os.path.dirname(os.path.realpath(__file__))
path = 'file:///' + path + '/myModelPath'
print(50*'*')
print('The path is: ' + path)
print(50*'*')
model.save(sc, path)
sameModel = MatrixFactorizationModel.load(sc, path)
