import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import RobustScaler

from stacking_regression_models import Constants


def read_data():
    train = pd.read_csv(Constants.DATA_TRAIN_CSV)
    test = pd.read_csv(Constants.DATA_TEST_CSV)
    return train, test


def prepare_data(train, test):
    y_train = train['y'].values
    id_test = test['ID']
    num_train = len(train)
    df_all = pd.concat([train, test])
    df_all.drop(['ID', 'y'], axis=1, inplace=True)
    # One-hot encoding of categorical/strings
    df_all = pd.get_dummies(df_all, drop_first=True)
    # Sscaling features
    scaler = RobustScaler()
    df_all = scaler.fit_transform(df_all)
    train = df_all[:num_train]
    test = df_all[num_train:]
    # Keep only the most contributing features
    sfm = SelectFromModel(LassoCV())
    sfm.fit(train, y_train)
    train = sfm.transform(train)
    test = sfm.transform(test)
    return train, y_train, test, id_test
