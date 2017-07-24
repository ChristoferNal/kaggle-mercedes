# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

import DataPreprocess
import NNFactory
from Constants import *


def fit_and_predict(samples_train_start, samples_train_stop, samples_predict_start, samples_predict_stop,
                    num_of_features, test, id_test, train, y_train):
    deep_bidirectional = NNFactory.get_deep_bidirectional(num_of_features)
    bidirectional = NNFactory.get_bidirectional(num_of_features)
    bidirectional_no_conv = NNFactory.get_bidirectional_no_conv(num_of_features)
    lgbr = lgb.LGBMRegressor(nthread=3, silent=True, learning_rate=0.05, max_depth=3,
                             colsample_bytree=0.8, n_estimators=100, subsample=0.9,
                             seed=777)
    xgbr = xgb.sklearn.XGBRegressor(max_depth=4, learning_rate=0.005, subsample=0.9,
                                    objective='reg:linear', n_estimators=1000)
    extra_trees = ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35)
    rf = RandomForestRegressor(n_estimators=250, n_jobs=2, max_depth=3, min_samples_split=25, min_samples_leaf=35)

    fit_models(bidirectional, bidirectional_no_conv, deep_bidirectional, extra_trees, lgbr, rf, xgbr,
               samples_train_start,
               samples_train_stop, train, y_train)

    deep_bidirectional_pred_x, bidirectional_pred_x, bidirectional_no_conv_pred_x, lgbr_x, xgbr_x, extra_trees_x, rf_x, \
    deep_bidirectional_pred_y, bidirectional_pred_y, bidirectional_no_conv_pred_y, lgbr_y, xgbr_y, extra_trees_y, rf_y \
        = predict(
        bidirectional, bidirectional_no_conv, deep_bidirectional, extra_trees, lgbr, rf, samples_predict_start,
        samples_predict_stop, xgbr, test, train)

    return create_dataframes(deep_bidirectional_pred_x, bidirectional_pred_x, bidirectional_no_conv_pred_x, lgbr_x,
                             xgbr_x,
                             extra_trees_x, rf_x, deep_bidirectional_pred_y, bidirectional_pred_y,
                             bidirectional_no_conv_pred_y, lgbr_y, xgbr_y, extra_trees_y, rf_y, test, id_test)


def create_dataframes(deep_bidirectional_pred_x, bidirectional_pred_x, bidirectional_no_conv_pred_x, lgbr_x, xgbr_x,
                      extra_trees_x, rf_x, deep_bidirectional_pred_y, bidirectional_pred_y,
                      bidirectional_no_conv_pred_y, lgbr_y, xgbr_y, extra_trees_y, rf_y, test, id_test):
    df_deep_bidirectional_pred_x = pd.DataFrame({'x': deep_bidirectional_pred_x})
    df_deep_bidirectional_pred_y = pd.DataFrame({'ID': id_test, 'y': deep_bidirectional_pred_y})
    df_bidirectional_pred_x = pd.DataFrame({'x': bidirectional_pred_x})
    df_bidirectional_pred_y = pd.DataFrame({'ID': id_test, 'y': bidirectional_pred_y})
    df_bidirectional_no_conv_pred_x = pd.DataFrame({'x': bidirectional_no_conv_pred_x})
    df_bidirectional_no_conv_pred_y = pd.DataFrame({'ID': id_test, 'y': bidirectional_no_conv_pred_y})
    df_lgbr_x = pd.DataFrame({'x': lgbr_x})
    df_lgbr_y = pd.DataFrame({'ID': id_test, 'y': lgbr_y})
    df_xgbr_x = pd.DataFrame({'x': xgbr_x})
    df_xgbr_y = pd.DataFrame({'ID': id_test, 'y': xgbr_y})
    df_extra_trees_x = pd.DataFrame({'x': extra_trees_x})
    df_extra_trees_y = pd.DataFrame({'ID': id_test, 'y': extra_trees_y})
    df_rf_x = pd.DataFrame({'x': rf_x})
    df_rf_y = pd.DataFrame({'ID': id_test, 'y': rf_y})

    return df_deep_bidirectional_pred_x, df_deep_bidirectional_pred_y, df_bidirectional_pred_x, df_bidirectional_pred_y, \
           df_bidirectional_no_conv_pred_x, df_bidirectional_no_conv_pred_y, df_lgbr_x, df_lgbr_y, df_xgbr_x, df_xgbr_y, \
           df_extra_trees_x, df_extra_trees_y, df_rf_x, df_rf_y


def predict(bidirectional, bidirectional_no_conv, deep_bidirectional, extra_trees, lgbr, rf, samples_predict_start,
            samples_predict_stop, xgbr, test, train):
    deep_bidirectional_pred_x = deep_bidirectional.predict(train[samples_predict_start:samples_predict_stop]).flatten()
    deep_bidirectional_pred_y = deep_bidirectional.predict(test).flatten()
    bidirectional_pred_x = bidirectional.predict(train[samples_predict_start:samples_predict_stop]).flatten()
    bidirectional_pred_y = bidirectional.predict(test).flatten()
    bidirectional_no_conv_pred_x = bidirectional_no_conv.predict(
        train[samples_predict_start:samples_predict_stop]).flatten()
    bidirectional_no_conv_pred_y = bidirectional_no_conv.predict(test).flatten()
    lgbr_x = lgbr.predict(train[samples_predict_start:samples_predict_stop]).flatten()
    lgbr_y = lgbr.predict(test).flatten()
    xgbr_x = xgbr.predict(train[samples_predict_start:samples_predict_stop]).flatten()
    xgbr_y = xgbr.predict(test).flatten()
    extra_trees_x = extra_trees.predict(train[samples_predict_start:samples_predict_stop]).flatten()
    extra_trees_y = extra_trees.predict(test).flatten()
    rf_x = rf.predict(train[samples_predict_start:samples_predict_stop]).flatten()
    rf_y = rf.predict(test).flatten()
    return deep_bidirectional_pred_x, bidirectional_pred_x, bidirectional_no_conv_pred_x, lgbr_x, xgbr_x, extra_trees_x, rf_x, deep_bidirectional_pred_y, bidirectional_pred_y, bidirectional_no_conv_pred_y, lgbr_y, xgbr_y, extra_trees_y, rf_y


def fit_models(bidirectional, bidirectional_no_conv, deep_bidirectional, extra_trees, lgbr, rf, xgbr,
               samples_train_start,
               samples_train_stop, train, y_train):
    deep_bidirectional.fit(train[samples_train_start:samples_train_stop],
                           y_train[samples_train_start:samples_train_stop], epochs=45,
                           verbose=2)
    bidirectional.fit(train[samples_train_start:samples_train_stop], y_train[samples_train_start:samples_train_stop],
                      epochs=100,
                      verbose=2)
    bidirectional_no_conv.fit(train[samples_train_start:samples_train_stop],
                              y_train[samples_train_start:samples_train_stop], epochs=75,
                              verbose=2)
    lgbr.fit(train[samples_train_start:samples_train_stop], y_train[samples_train_start:samples_train_stop])
    xgbr.fit(train[samples_train_start:samples_train_stop], y_train[samples_train_start:samples_train_stop])
    extra_trees.fit(train[samples_train_start:samples_train_stop], y_train[samples_train_start:samples_train_stop])
    rf.fit(train[samples_train_start:samples_train_stop], y_train[samples_train_start:samples_train_stop])


def save_results(df_deep_bidirectional_pred_x, df_deep_bidirectional_pred_y, df_bidirectional_pred_x,
                 df_bidirectional_pred_y, \
                 df_bidirectional_no_conv_pred_x, df_bidirectional_no_conv_pred_y, df_lgbr_x, df_lgbr_y, df_xgbr_x,
                 df_xgbr_y, \
                 df_extra_trees_x, df_extra_trees_y, df_rf_x, df_rf_y):
    df_deep_bidirectional_pred_x.to_csv(DEEP_BIDIRECTIONAL_PRED_X_CSV, index=False)
    df_deep_bidirectional_pred_y.to_csv(DEEP_BIDIRECTIONAL_PRED_Y_CSV, index=False)
    df_bidirectional_pred_x.to_csv(BIDIRECTIONAL_PRED_X_CSV, index=False)
    df_bidirectional_pred_y.to_csv(BIDIRECTIONAL_PRED_Y_CSV, index=False)
    df_bidirectional_no_conv_pred_x.to_csv(NO_CONV_PRED_X_CSV, index=False)
    df_bidirectional_no_conv_pred_y.to_csv(NO_CONV_PRED_Y_CSV, index=False)
    df_lgbr_x.to_csv(DF_LGBR_X_CSV, index=False)
    df_lgbr_y.to_csv(DF_LGBR_Y_CSV, index=False)
    df_xgbr_x.to_csv(DF_XGBR_X_CSV, index=False)
    df_xgbr_y.to_csv(DF_XGBR_Y_CSV, index=False)
    df_extra_trees_x.to_csv(DF_EXTRA_TREES_X_CSV, index=False)
    df_extra_trees_y.to_csv(DF_EXTRA_TREES_Y_CSV, index=False)
    df_rf_x.to_csv(DF_RF_X_CSV, index=False)
    df_rf_y.to_csv(DF_RF_Y_CSV, index=False)


def save_results2(df_deep_bidirectional_pred_x, df_deep_bidirectional_pred_y, df_bidirectional_pred_x,
                  df_bidirectional_pred_y, \
                  df_bidirectional_no_conv_pred_x, df_bidirectional_no_conv_pred_y, df_lgbr_x, df_lgbr_y, df_xgbr_x,
                  df_xgbr_y, \
                  df_extra_trees_x, df_extra_trees_y, df_rf_x, df_rf_y):
    df_deep_bidirectional_pred_x.to_csv(DEEP_BIDIRECTIONAL_PRED_X2_CSV, index=False)
    df_deep_bidirectional_pred_y.to_csv(DEEP_BIDIRECTIONAL_PRED_Y2_CSV, index=False)
    df_bidirectional_pred_x.to_csv(BIDIRECTIONAL_PRED_X2_CSV, index=False)
    df_bidirectional_pred_y.to_csv(BIDIRECTIONAL_PRED_Y2_CSV, index=False)
    df_bidirectional_no_conv_pred_x.to_csv(NO_CONV_PRED_X2_CSV, index=False)
    df_bidirectional_no_conv_pred_y.to_csv(NO_CONV_PRED_Y2_CSV, index=False)
    df_lgbr_x.to_csv(DF_LGBR_X2_CSV, index=False)
    df_lgbr_y.to_csv(DF_LGBR_Y2_CSV, index=False)
    df_xgbr_x.to_csv(DF_XGBR_X2_CSV, index=False)
    df_xgbr_y.to_csv(DF_XGBR_Y2_CSV, index=False)
    df_extra_trees_x.to_csv(DF_EXTRA_TREES_X2_CSV, index=False)
    df_extra_trees_y.to_csv(DF_EXTRA_TREES_Y2_CSV, index=False)
    df_rf_x.to_csv(DF_RF_X2_CSV, index=False)
    df_rf_y.to_csv(DF_RF_Y2_CSV, index=False)


def run():
    train, test = DataPreprocess.read_data()
    train, y_train, test, id_test = DataPreprocess.prepare_data(train, test)
    num_of_features = train.shape[1]
    print ('Number of features : %d' % num_of_features)
    samples_train_start = 0
    samples_train_stop = 1800
    samples_predict_start = 1800
    samples_predict_stop = 4209

    df_deep_bidirectional_pred_x, df_deep_bidirectional_pred_y, df_bidirectional_pred_x, df_bidirectional_pred_y, \
    df_bidirectional_no_conv_pred_x, df_bidirectional_no_conv_pred_y, df_lgbr_x, df_lgbr_y, df_xgbr_x, df_xgbr_y, \
    df_extra_trees_x, df_extra_trees_y, df_rf_x, df_rf_y = fit_and_predict(samples_train_start, samples_train_stop,
                                                                           samples_predict_start,
                                                                           samples_predict_stop, num_of_features,
                                                                           test[
                                                                           samples_predict_start:samples_predict_stop],
                                                                           id_test[
                                                                           samples_predict_start:samples_predict_stop],
                                                                           train, y_train)
    save_results(df_deep_bidirectional_pred_x, df_deep_bidirectional_pred_y, df_bidirectional_pred_x,
                 df_bidirectional_pred_y, \
                 df_bidirectional_no_conv_pred_x, df_bidirectional_no_conv_pred_y, df_lgbr_x, df_lgbr_y, df_xgbr_x,
                 df_xgbr_y, \
                 df_extra_trees_x, df_extra_trees_y, df_rf_x, df_rf_y)

    samples_train_start = 1800
    samples_train_stop = 4209
    samples_predict_start = 0
    samples_predict_stop = 1800
    df_deep_bidirectional_pred_x, df_deep_bidirectional_pred_y, df_bidirectional_pred_x, df_bidirectional_pred_y, \
    df_bidirectional_no_conv_pred_x, df_bidirectional_no_conv_pred_y, df_lgbr_x, df_lgbr_y, df_xgbr_x, df_xgbr_y, \
    df_extra_trees_x, df_extra_trees_y, df_rf_x, df_rf_y = fit_and_predict(samples_train_start, samples_train_stop,
                                                                           samples_predict_start,
                                                                           samples_predict_stop, num_of_features,
                                                                           test[
                                                                           samples_predict_start:samples_predict_stop],
                                                                           id_test[
                                                                           samples_predict_start:samples_predict_stop],
                                                                           train, y_train)
    save_results2(df_deep_bidirectional_pred_x, df_deep_bidirectional_pred_y, df_bidirectional_pred_x,
                  df_bidirectional_pred_y, \
                  df_bidirectional_no_conv_pred_x, df_bidirectional_no_conv_pred_y, df_lgbr_x, df_lgbr_y, df_xgbr_x,
                  df_xgbr_y, \
                  df_extra_trees_x, df_extra_trees_y, df_rf_x, df_rf_y)
