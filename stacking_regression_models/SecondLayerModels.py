import matplotlib.pyplot as plt
import numpy as np
from sklearn import clone
from sklearn import linear_model
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from Constants import *
from stacking_regression_models import Constants


class StackingCVRegressorAveraged(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors, meta_regressor, n_folds=5):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds

    def fit(self, X, y):
        self.regr_ = [list() for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        for i, clf in enumerate(self.regressors):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(clf)
                self.regr_[i].append(instance)

                instance.fit(X[train_idx], y[train_idx])
                y_pred = instance.predict(X[holdout_idx])
                out_of_fold_predictions[holdout_idx, i] = y_pred

        self.meta_regr_.fit(out_of_fold_predictions, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
                                            np.column_stack([r.predict(X) for r in regrs]).mean(axis=1)
                                            for regrs in self.regr_
                                            ])
        return self.meta_regr_.predict(meta_features)


class StackingCVRegressorRetrained(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors, meta_regressor, n_folds=5, use_features_in_secondary=False):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds
        self.use_features_in_secondary = use_features_in_secondary

    def fit(self, X, y):
        self.regr_ = [clone(x) for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        # Create out-of-fold predictions for training meta-model
        for i, regr in enumerate(self.regr_):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(regr)
                instance.fit(X[train_idx], y[train_idx])
                out_of_fold_predictions[holdout_idx, i] = instance.predict(X[holdout_idx])

        # Train meta-model
        if self.use_features_in_secondary:
            self.meta_regr_.fit(np.hstack((X, out_of_fold_predictions)), y)
        else:
            self.meta_regr_.fit(out_of_fold_predictions, y)

        # Retrain base models on all data
        for regr in self.regr_:
            regr.fit(X, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
                                            regr.predict(X) for regr in self.regr_
                                            ])

        if self.use_features_in_secondary:
            return self.meta_regr_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_regr_.predict(meta_features)


class AveragingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors):
        self.regressors = regressors

    def fit(self, X, y):
        self.regr_ = [clone(x) for x in self.regressors]

        # Train base models
        for regr in self.regr_:
            regr.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
                                          regr.predict(X) for regr in self.regr_
                                          ])
        return np.mean(predictions, axis=1)


def run():
    y = pd.read_csv(DF_RF_Y_CSV)
    x = pd.read_csv(DF_RF_X_CSV)
    y1 = pd.read_csv(DF_EXTRA_TREES_Y_CSV)
    x1 = pd.read_csv(DF_EXTRA_TREES_X_CSV)
    y2 = pd.read_csv(DF_XGBR_Y_CSV)
    x2 = pd.read_csv(DF_XGBR_X_CSV)
    y3 = pd.read_csv(DF_LGBR_Y_CSV)
    x3 = pd.read_csv(DF_LGBR_X_CSV)
    y4 = pd.read_csv(NO_CONV_PRED_Y_CSV)
    x4 = pd.read_csv(NO_CONV_PRED_X_CSV)
    y5 = pd.read_csv(BIDIRECTIONAL_PRED_Y_CSV)
    x5 = pd.read_csv(BIDIRECTIONAL_PRED_X_CSV)
    y6 = pd.read_csv(DEEP_BIDIRECTIONAL_PRED_Y_CSV)
    x6 = pd.read_csv(DEEP_BIDIRECTIONAL_PRED_X_CSV)
    yy = pd.read_csv(DF_RF_Y2_CSV)
    xx = pd.read_csv(DF_RF_X2_CSV)
    yy1 = pd.read_csv(DF_EXTRA_TREES_Y2_CSV)
    xx1 = pd.read_csv(DF_EXTRA_TREES_X2_CSV)
    yy2 = pd.read_csv(DF_XGBR_Y2_CSV)
    xx2 = pd.read_csv(DF_XGBR_X2_CSV)
    yy3 = pd.read_csv(DF_LGBR_Y2_CSV)
    xx3 = pd.read_csv(DF_LGBR_X2_CSV)
    yy4 = pd.read_csv(NO_CONV_PRED_Y2_CSV)
    xx4 = pd.read_csv(NO_CONV_PRED_X2_CSV)
    yy5 = pd.read_csv(BIDIRECTIONAL_PRED_Y2_CSV)
    xx5 = pd.read_csv(BIDIRECTIONAL_PRED_X2_CSV)
    yy6 = pd.read_csv(DEEP_BIDIRECTIONAL_PRED_Y2_CSV)
    xx6 = pd.read_csv(DEEP_BIDIRECTIONAL_PRED_X2_CSV)
    train = pd.read_csv(Constants.DATA_TRAIN_CSV)
    y_train = train['y'][:].values
    x_train0 = pd.concat([x, x1, x3, x4, x5, x6], axis=1)
    x_final0 = pd.concat([y['y'], y1['y'], y3['y'], y4['y'], y5['y'], y6['y']], axis=1)
    xx_train = pd.concat([xx, xx1, xx3, xx4, xx5, xx6], axis=1)
    xx_final = pd.concat([yy['y'], yy1['y'], yy3['y'], yy4['y'], yy5['y'], yy6['y']], axis=1)
    test = pd.read_csv(Constants.DATA_TEST_CSV)
    id_test = test['ID']
    x_train = pd.concat([xx_train, x_train0], axis=0).values
    x_final = pd.concat([xx_final, x_final0], axis=0).values
    y_mean = np.mean(y_train)
    en = make_pipeline(RobustScaler(), SelectFromModel(Lasso(alpha=0.03)), ElasticNet(alpha=0.001, l1_ratio=0.1))
    rf = RandomForestRegressor(n_estimators=250, n_jobs=2, max_depth=3, min_samples_split=25, min_samples_leaf=35)
    et = ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35)
    xgbm = xgb.sklearn.XGBRegressor(max_depth=4, learning_rate=0.005, subsample=0.9, base_score=y_mean,
                                    objective='reg:linear', n_estimators=1000)
    lgbm = lgb.LGBMRegressor(nthread=3, silent=True, learning_rate=0.05, max_depth=3,
                             colsample_bytree=0.8, n_estimators=100, subsample=0.9,
                             seed=777)
    mpl = MLPRegressor(hidden_layer_sizes=1000)
    lasso = linear_model.Lasso(alpha=13)
    stack_avg = StackingCVRegressorAveraged((en, rf, et, lgbm), ElasticNet(l1_ratio=0.1, alpha=1.4))
    #
    stack_with_feats = StackingCVRegressorRetrained((en, rf, et, lgbm), et, use_features_in_secondary=True)
    #
    stack_retrain = StackingCVRegressorRetrained((en, rf, et, lgbm), ElasticNet(l1_ratio=0.1, alpha=1.4))
    # averaged = AveragingRegressor((en, rf, et, xgbm, lgbm))
    averaged = AveragingRegressor((stack_avg, stack_with_feats, stack_retrain))
    results = cross_val_score(en, x_train, y_train, cv=5, scoring='r2')
    print("ElasticNet score: %.4f (%.4f)" % (results.mean(), results.std()))
    results = cross_val_score(rf, x_train, y_train, cv=5, scoring='r2')
    print("RandomForest score: %.4f (%.4f)" % (results.mean(), results.std()))
    results = cross_val_score(et, x_train, y_train, cv=5, scoring='r2')
    print("ExtraTrees score: %.4f (%.4f)" % (results.mean(), results.std()))
    results = cross_val_score(xgbm, x_train, y_train, cv=5, scoring='r2')
    print("XGBoost score: %.4f (%.4f)" % (results.mean(), results.std()))
    results = cross_val_score(lgbm, x_train, y_train, cv=5, scoring='r2')
    print("LGBM score: %.4f (%.4f)" % (results.mean(), results.std()))
    results = cross_val_score(mpl, x_train, y_train, cv=5, scoring='r2')
    print("MPL base models score: %.4f (%.4f)" % (results.mean(), results.std()))
    results = cross_val_score(lasso, x_train, y_train, cv=5, scoring='r2')
    print("Lasso base models score: %.4f (%.4f)" % (results.mean(), results.std()))
    results = cross_val_score(stack_with_feats, x_train, y_train, cv=5, scoring='r2')
    print("Stacking (with primary feats) score: %.4f (%.4f)" % (results.mean(), results.std()))
    results = cross_val_score(stack_retrain, x_train, y_train, cv=5, scoring='r2')
    print("Stacking (retrained) score: %.4f (%.4f)" % (results.mean(), results.std()))
    results = cross_val_score(stack_avg, x_train, y_train, cv=5, scoring='r2')
    print("Stacking (averaged) score: %.4f (%.4f)" % (results.mean(), results.std()))
    results = cross_val_score(averaged, x_train, y_train, cv=5, scoring='r2')
    print("Averaged base models score: %.4f (%.4f)" % (results.mean(), results.std()))
    averaged = AveragingRegressor((en, rf, et, xgbm, lgbm, mpl))
    averaged.fit(x_train, y_train)
    y_final = averaged.predict(x_final)
    df_sub = pd.DataFrame({'ID': id_test, 'y': y_final})
    df_sub.to_csv(Constants.STACKING_RESULTS_CSV, index=False)
