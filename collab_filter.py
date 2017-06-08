##########################################
df = df.withColumn('userId', df['userId'].cast(IntegerType()))
df = df.withColumn('movieId', df['movieId'].cast(IntegerType()))
df = df.withColumn('rating', df['rating'].cast(DoubleType()))
df = df.withColumn('timestamp', df['timestamp'].cast(IntegerType()))
(training, test) = df.randomSplit([0.8, 0.2])
model = als.fit(training)
predictions = model.transform(test)
predictions = predictions.withColumn('rating', predictions['rating'].cast(FloatType()))
print [f.dataType for f in predictions.schema.fields]

predictions.where(predictions.prediction.isNull()).count()

evaluator = RegressionEvaluator(metricName="mae", labelCol="rating")
rmse = evaluator.evaluate(predictions)
##########################################