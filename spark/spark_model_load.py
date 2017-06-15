import re
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
output = spark.read.csv('./output_recommender.csv', header=True)

# Recommend n products for all users.
products_for_users = bestModel.recommendProductsForUsers(3).collect()
print len(products_for_users)

# Recommend n products for each user loop.
test = orders.filter(orders['eval_set'] == 'test')
test_ids = test.select('user_id')
test_list = test_ids.rdd.map(lambda x: x[0]).collect()
# test_list = test_list[-100:]
rec_ = []
for i, user_id in enumerate(test_list):
    print 'Round: ', i
    if i % 100 == 0:
        pd.DataFrame(rec_).to_csv('rec_{}.csv'.format(i), index=False)
        rec_ = []
        print 'Output rec_: ', rec_, ' | ', i
    rec_.append(bestModel.recommendProducts(int(user_id), 8))

# Read csv file, clean up, and concat
def load_file(file_):
    df = pd.read_csv('rec_{}.csv'.format(file_))
    df = df.drop('Unnamed: 0', axis=1)
    # Define a find user function.
    def find_user(row_, col_='0'):
        return int(re.findall(r"([\d.]*\d+)", df[col_][row_])[0])
    user_list = map(find_user, range(0, df.shape[0]))
    df['user'] = user_list
    # Define a find product function.
    def find_product(row_, col_):
        return int(re.findall(r"([\d.]*\d+)", df[col_][row_])[1])
    for i in range(0, 8):
        prod_list = map(find_product, range(0, df.shape[0]), [str(i)]*100)
        df[i] = prod_list
        df = df.drop(str(i), axis=1)
    # Join the product columns into a list, then drop the columns.
    df['products'] = df[df.columns[1:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
    df = df[:][['user', 'products']]
    return df

df_list = map(load_file, range(100, 75100, 100))
for i, df in enumerate(df_list):
    if i == 0:
        output = pd.DataFrame(df, columns=df.columns)
    else:
        output = pd.concat([output, df], axis=0)

# Read the output file, and join with test dataframe to get the order_id. Make submission file.
output.to_csv('output_recommender.csv', index=False)
output_rec = spark.read.csv('./output_recommender.csv', header=True)
submission = output_rec.join(test, output_rec.user == test.user_id, 'left').select('order_id', 'products')
submission = pd.DataFrame(submission)
submission.columns = ['order_id', 'products']
submission.to_csv('spark_rec_submission_1.csv', index=False)
