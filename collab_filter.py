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
