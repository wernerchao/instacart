import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
# from pyspark.ml.recommendation import ALS


orders = spark.read.csv('./data/orders.csv', header=True)
train_orders = spark.read.csv('./order_products__train.csv', header=True)
prior_orders = spark.read.csv('./order_products__prior.csv', header=True)
products = spark.read.csv('./products.csv', header=True)
aisles = spark.read.csv('./aisles.csv', header=True)
departments = spark.read.csv('./departments.csv', header=True)

orders.cache()
prior_orders.cache()

print pd.DataFrame(orders.take(5), columns=orders.columns).transpose()
print orders.show(5)
print orders.describe().toPandas().transpose()
print orders.select('eval_set').distinct().show() # show the unique values in 'eval_set' column
print pd.DataFrame(orders.filter(orders['eval_set'] == 'test').take(12), columns=orders.columns).transpose() # show the 'test' rows


###########################################
### Prepare for Collaborating Filtering ###
###########################################

# Join orders to prior_orders on order_id. This is training set.
prior_orders = prior_orders.join(orders, prior_orders.order_id == orders.order_id, 'left') \
                    .select(orders.user_id, prior_orders.order_id, prior_orders.product_id, \
                            prior_orders.add_to_cart_order, prior_orders.reordered)
print pd.DataFrame(prior_orders.take(5), columns=prior_orders.columns).transpose()
prior_orders = prior_orders.withColumn('add_to_cart_order', prior_orders['add_to_cart_order'].cast(FloatType()))
prior_orders = prior_orders.withColumn('rank', F.pow(F.col('add_to_cart_order'), -1))
prior_orders = prior_orders.withColumn('rating', F.col('rank')*0.5+F.col('reordered')*0.5) # Creates a target variable called rating

# Join orders to train_orders on order_id. This is validation set.
train_orders = train_orders.join(orders, train_orders.order_id == orders.order_id, 'left') \
                    .select(orders.user_id, train_orders.order_id, train_orders.product_id, \
                            train_orders.add_to_cart_order, train_orders.reordered)
print pd.DataFrame(train_orders.take(5), columns=train_orders.columns).transpose()
train_orders = train_orders.withColumn('add_to_cart_order', F.col('add_to_cart_order').cast(FloatType()))
train_orders = train_orders.withColumn('rank', F.pow(F.col('add_to_cart_order'), -1))
train_orders = train_orders.withColumn('rating', F.col('rank')*0.5+F.col('reordered')*0.5)

# Make training, validation set
training = prior_orders.select(['user_id', 'product_id', 'rating'])
valid = train_orders.select(['user_id', 'product_id', 'rating'])
training = training.withColumn('user_id', F.col('user_id').cast(IntegerType()))
training = training.withColumn('product_id', F.col('product_id').cast(IntegerType()))
print [f.dataType for f in training.schema.fields]
valid = valid.withColumn('user_id', F.col('user_id').cast(IntegerType()))
valid = valid.withColumn('product_id', F.col('product_id').cast(IntegerType()))
print [f.dataType for f in valid.schema.fields]

# Training with old MlLib
model = ALS.train(training, rank=10, iterations=10)
model.save(sc, './model_1')
test = orders.filter(orders['eval_set'] == 'test')
test_id = test.select('user_id')
test_id_list = test_id.collect()
def unionAll(*dfs):
    ''' Stack the DataFrames vertically '''
    return reduce(F.DataFrame.unionAll, dfs)
product_all = unionAll(training.select('product_id'), valid.select('product_id'))

# Recommend product for users in test set
rec = test_id.rdd.map(lambda row: model.recommendProducts(row, 3))
rec_df = test.withColumn('rec', model.recommendProducts(test_id_list, 3))

make_rec = F.udf(lambda row: model.recommendProducts(row, 3))
rec_df = test.withColumn('rec', make_rec(F.col('user_id')))
make_rec(1, 3) # Recommend user 1, 3 products


# Training...and predicting...
als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="product_id", ratingCol="rating")
model = als.fit(training)
valid_pred = model.transform(valid)
print valid_pred.filter(valid_pred.prediction.isNotNull()).count()
print valid_pred.filter(valid_pred.rating.isNotNull()).count()
valid_pred = valid_pred.na.drop(subset=["prediction"])
print valid_pred.count()
# Save prediction
print type(valid_pred)
valid_pred.write.csv('./valid_pred_2.csv', header=True)
# Evaluate RMSE: 0.28033986860868765
evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating')
rmse_1 = evaluator.evaluate(valid_pred)
print "RMSE: ", rmse_1

# Training with implicit feedbacks
als_imp = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="product_id", ratingCol="rating", implicitPrefs=True)
model_imp = als_imp.fit(training)
valid_pred_imp = model_imp.transform(valid)
print valid_pred_imp.filter(valid_pred_imp.prediction.isNotNull()).count()
print valid_pred_imp.filter(valid_pred_imp.rating.isNotNull()).count()
valid_pred_imp = valid_pred_imp.na.drop(subset=["prediction"])
print valid_pred_imp.count()
# Save prediction
valid_pred_imp.write.csv('./valid_pred_imp_2.csv', header=True)
# Evaluate RMSE: 0.49590265732641114
evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating')
rmse_2 = evaluator.evaluate(valid_pred_imp)
print "RMSE: ", rmse_2


################################################
### Basic Data Manipulation from Nick Sarris ###
################################################

# Get test set, and remove user_id in orders that's not in the test set
test = orders.filter(orders['eval_set'] == 'test')
test_ids = test.select('user_id')
test_list = test_ids.rdd.map(lambda x: x[0]).collect()
orders_filt = orders.user_id.isin(test_list)
orders = orders[orders_filt] # shape = (1242497, )

# Combine prior rows by user_id, add product_ids to a list
print pd.DataFrame(prior_orders.take(5), columns=prior.columns).transpose()
prior_products = prior_orders.groupby('order_id').agg(F.collect_list('product_id'))
# prior_products = prior.groupby('order_id').agg(F.collect_list(F.struct('product_id')))

# Combine train rows by user_id, add product_ids to a list
train_products = train_orders.groupby('order_id').agg(F.collect_list('product_id'))

# Seperate orders into prior/train sets
# turns out there are no test user_ids in the training set so train will be empty
prior = orders.filter(orders['eval_set'] == 'prior')
train = orders.filter(orders['eval_set'] == 'train')

# Find the number of the last order placed
prior.withColumn('num_orders', prior.groupby('user_id').agg(F.max('order_number')))


#####################################
### SQL Join and Concat DataFrame ###
#####################################

# Concat train and prior df vertically
def unionAll(*dfs):
    ''' Stack the DataFrames vertically '''
    return reduce(F.DataFrame.unionAll, dfs)
train_prior = unionAll(order_products__train, order_products__prior)
train_prior.describe().toPandas().transpose()

# Join orders table, train table, and prior table
orders.createOrReplaceTempView('orders_table')
train_prior.createOrReplaceTempView('train_prior_table')
orders_train_prior = \
    spark.sql('SELECT * FROM orders_table o \
                LEFT JOIN train_prior_table t ON o.order_id = t.order_id')

pd.DataFrame(orders_train_prior.take(20), columns=orders_train_prior.columns).transpose() # shows the new dataframe

# Shows df with eval_set = test, train, or prior
pd.DataFrame(orders_train_prior.filter(orders_train_prior['eval_set'] == 'test').take(5), \
            columns=orders_train_prior.columns).transpose()

