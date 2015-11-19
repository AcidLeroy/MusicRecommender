from pyspark import SparkContext, SparkConf

conf = SparkConf() \
    .setAppName("Collaborative Filter") \
    .set("spark.executor.memory", "5g")
sc = SparkContext(conf=conf)

data = sc.textFile('file:////home/vj/Desktop/MusicRecommender/src/test.txt')
dataKV = data.map(lambda x: (x.split(" ")[0], x))
userPlays = data.map(lambda x: (x.split(" ")[0], float(x.split(" ")[2])))
userMax   = userPlays.foldByKey(0,max)
userJoin = dataKV.join(userMax)
Ndata = userJoin.map(lambda x: (x[0], x[1][0].split(" ")[1], 5*float(x[1][0].split(" ")[2])/x[1][1]))
Ndata = Ndata.map(lambda x: x[0] + ' ' + x[1] + ' ' + str(round(x[2])))
Ndata.saveAsTextFile('file:///home/vj/Desktop/MusicRecommender/src/NormalizedFiles')
