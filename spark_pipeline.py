from pyspark.sql.functions import *

orders = spark.read.csv('./orders.csv', header=True)
train = spark.read.csv('./order_products__train.csv', header=True)
prior = spark.read.csv('./order_products__prior.csv', header=True)
products = spark.read.csv('./products.csv', header=True)
aisles = spark.read.csv('./aisles.csv', header=True)
departments = spark.read.csv('./departments.csv', header=True)

orders = orders.cache()
train = traom.cache()
prior = prior.cache()
products = products.cache()
aisles = aisles.cache()
departments = departments.cache()

print pd.DataFrame(orders.take(5), columns=orders.columns).transpose()
print orders.show(5)
print orders.describe().toPandas().transpose()
print orders.select('eval_set').distinct().show() # show the unique values in 'eval_set' column
print pd.DataFrame(orders.filter(orders['eval_set'] == 'test').take(12), columns=orders.columns).transpose() # show the 'test' rows

# Get test set, and remove user_id in orders that's not in the test set
test = orders.filter(orders['eval_set'] == 'test')
test_ids = test.select('user_id')
test_list = test_ids.rdd.map(lambda x: x[0]).collect()
orders_filt = orders.user_id.isin(test_list)
orders = orders[orders_filt] # shape = (1242497, )




# Concat train and prior df vertically
def unionAll(*dfs):
    ''' Stack the DataFrames vertically '''
    return reduce(DataFrame.unionAll, dfs)
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

