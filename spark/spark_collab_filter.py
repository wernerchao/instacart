from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

df = spark.read.csv('ratings.csv', header=True)
df = df.withColumn('userId', df['userId'].cast(IntegerType()))
df = df.withColumn('movieId', df['movieId'].cast(IntegerType()))
df = df.withColumn('rating', df['rating'].cast(DoubleType()))
df = df.withColumn('timestamp', df['timestamp'].cast(IntegerType()))
(training, test) = df.randomSplit([0.8, 0.2])

# Training...
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(training)
predictions = model.transform(test)
predictions = predictions.withColumn('rating', predictions['rating'].cast(FloatType()))
print [f.dataType for f in predictions.schema.fields]

print predictions.where(predictions.prediction.isNull()).count()

evaluator = RegressionEvaluator(metricName="mae", labelCol="rating")
rmse = evaluator.evaluate(predictions)
print 'RMSE: ', rmse


# Collect product feature matrix
productFeatures = bestModel.productFeatures().collect() 
productArray=[]
productFeaturesArray=[]
for x in productFeatures:
    productArray.append(x[0])
    productFeaturesArray.append(x[1])  
    matrix=np.matrix(productFeaturesArray)
    productArrayBroadCast=sc.broadcast(productArray)
    productFeaturesArraybroadcast=sc.broadcast(matrix.T)

def func(iterator):
    userFeaturesArray = []
    userArray = []
    for x in iterator:
        userArray.append(x[0])
        userFeaturesArray.append(x[1])
        userFeatureMatrix = np.matrix(userFeaturesArray)
        userRecommendationArray = userFeatureMatrix*(productFeaturesArraybroadcast.value)
        mappedUserRecommendationArray = []
        #Extract ratings from the matrix
        i=0
        for i in range(0,len(userArray)):
            ratingdict={}
            j=0
            for j in range(0,len(productArrayBroadcast.value)):
                ratingdict[str(productArrayBroadcast.value[j])]=userRecommendationArray.item((i,j))
                j=j+1
            #Take the top 8 recommendations for the user
            sort_apps=sorted(ratingdict.keys(), key=lambda x: x[1])[:8]
            sort_apps='|'.join(sort_apps)
            mappedUserRecommendationArray.append((userArray[i],sort_apps))
            i=i+1
    return [x for x in mappedUserRecommendationArray]


recommendations=model.userFeatures().repartition(2000).mapPartitions(func)