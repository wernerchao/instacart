import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
IDIR = '../../features/'

if __name__ == '__main__':
    print('Directly load features into xgboost')
    df_train = pd.read_csv(IDIR + 'df_train.csv')
    labels = pd.read_csv(IDIR + 'labels.csv')
    f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items', \
                'user_average_days_between_orders', 'user_average_basket', \
                'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio', \
                'aisle_id', 'department_id', 'product_orders', 'product_reorders', \
                'product_reorder_rate', 'UP_orders', 'UP_orders_ratio', \
                'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last', \
                'UP_delta_hour_vs_last']
    df_train = df_train[f_to_use]

    # eta = [0.1]
    params = {
    "objective"           : "reg:logistic",
    "eval_metric"         : "logloss",
    "eta"                 : 0.5,
    "max_depth"           : 6,
    "min_child_weight"    : 10,
    "gamma"               : 0.70,
    "subsample"           : 0.76,
    "colsample_bytree"    : 0.95,
    "alpha"               : 2e-05,
    "lambda"              : 10
    }
    ROUNDS = 100

    print('XGB train...')
    skf = StratifiedKFold(n_splits = 5)
    fold_count = 0
    for train_index, val_index in skf.split(df_train, labels['0']):
        X_train, X_val = df_train.iloc[train_index, :], df_train.iloc[val_index, :]
        y_train, y_val = labels.iloc[train_index, :], labels.iloc[val_index, :]

        print('formating for xgb')
        d_train = xgb.DMatrix(X_train, label = y_train)
        d_valid = xgb.DMatrix(X_val, label = y_val)
        watchlist = [(d_train, 'train'), (d_valid, 'val')]

        print('Training...')
        bst = xgb.train(params = params, \
                        dtrain = d_train, \
                        num_boost_round = ROUNDS, \
                        evals = watchlist, \
                        verbose_eval = 10, \
                        early_stopping_rounds = '10')
        d_test = xgb.DMatrix(X_val)
        pred_val = pd.DataFrame(bst.predict(d_test))
        pred_val.to_csv('./xgb_train_fold_{}.csv'.format(fold_count))
        pd.DataFrame(y_val).to_csv('./y_val_fold){}.csv'.format(fold_count))
        fold_count += 1
        break
