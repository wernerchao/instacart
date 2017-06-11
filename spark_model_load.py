import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel

# Load Trained Model.
valid_pred = spark.read.csv('./valid_pred_2.csv', header=True)
bestModel = MatrixFactorizationModel.load(sc, './model_1')
orders = spark.read.csv('./orders.csv', header=True)

# Recommend n products for each user loop.
test = orders.filter(orders['eval_set'] == 'test')
test_ids = test.select('user_id')
test_list = test_ids.rdd.map(lambda x: x[0]).collect()
rec_ = []
for user_id in test_list:
    rec_ = bestModel.recommendProducts(int(user_id), 3)



# Collect product feature matrix
productFeatures = bestModel.productFeatures().collect()
productArray = []
productFeaturesArray = []
for x in productFeatures:
    productArray.append(x[0])
    productFeaturesArray.append(x[1])
    matrix = np.matrix(productFeaturesArray)
    productArrayBroadCast = sc.broadcast(productArray)
    productFeaturesArraybroadcast = sc.broadcast(matrix.T)

def func(iterator):
    userFeaturesArray = []
    userArray = []
    for x in iterator:
        userArray.append(x[0])
        userFeaturesArray.append(x[1])
        userFeatureMatrix = np.matrix(userFeaturesArray)
        userRecommendationArray = userFeatureMatrix*(productFeaturesArraybroadcast.value)
        mappedUserRecommendationArray = []

        # Extract ratings from the matrix
        i=0
        for i in range(0,len(userArray)):
            ratingdict = {}
            j = 0

            for j in range(0,len(productArrayBroadcast.value)):
                ratingdict[str(productArrayBroadcast.value[j])] = userRecommendationArray.item((i, j))
                j = j + 1

            # Take the top 8 recommendations for the user
            sort_apps = sorted(ratingdict.keys(), key=lambda x: x[1])[:8]
            sort_apps = '|'.join(sort_apps)
            mappedUserRecommendationArray.append((userArray[i], sort_apps))
            i = i + 1
    return [x for x in mappedUserRecommendationArray]

recommendations = model.userFeatures().repartition(2000).mapPartitions(func)
