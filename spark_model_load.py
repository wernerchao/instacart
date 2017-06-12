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

# Recommend n products for all users.
products_for_users = bestModel.recommendProductsForUsers(3).collect()
print len(products_for_users)

# Recommend n products for each user loop.
test = orders.filter(orders['eval_set'] == 'test')
test_ids = test.select('user_id')
test_list = test_ids.rdd.map(lambda x: x[0]).collect()
rec_ = []
for i, user_id in enumerate(test_list):
    print 'Round: ', i
    if i % 100 == 0:
        pd.DataFrame(rec_).to_csv('rec_{}.csv'.format(i))
        rec_ = []
        print 'Output rec_: ', rec_, ' | ', i
    rec_.append(bestModel.recommendProducts(int(user_id), 8))


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
        userRecommendationArray = userFeatureMatrix * (productFeaturesArraybroadcast.value)
        mappedUserRecommendationArray = []
        # Extract ratings from the matrix
        i=0
        for i in range(0,len(userArray)):
            ratingdict={}
            j=0
            for j in range(0,len(productArrayBroadcast.value)):
                ratingdict[str(productArrayBroadcast.value[j])]=userRecommendationArray.item((i,j))
                j=j+1
            # Take the top 8 recommendations for the user
            sort_apps=sorted(ratingdict.keys(), key=lambda x: x[1])[:8]
            sort_apps='|'.join(sort_apps)
            mappedUserRecommendationArray.append((userArray[i],sort_apps))
            i=i+1
        yield [x for x in mappedUserRecommendationArray]

recommendations = bestModel.userFeatures().repartition(2000).mapPartitions(func)
