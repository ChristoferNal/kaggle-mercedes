import numpy as np
from sklearn.model_selection import StratifiedKFold

import NNFactory


def run_cv(train, y_train, epochs = 33, mode=0):
    if mode > 2:
        return
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    cvscores = []
    for train_k, test_k in kfold.split(train, y_train):
        if mode == 0:
            model = NNFactory.get_deep_bidirectional()
        elif mode == 1:
            model = NNFactory.get_bidirectional()
        elif mode == 2:
            model = NNFactory.get_bidirectional_no_conv()
        model.fit(train[train_k], y_train[train_k], epochs=epochs, verbose=2)
        scores = model.evaluate(train[test_k], y_train[test_k], verbose=2)
        print('- - - - - -  - -  - - -  - - - -  - - - -  - - - - - -  - -')
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        print('- - - - - -  - -  - - -  - - - -  - - - -  - - - - - -  - -')
        cvscores.append(scores[1] * 100)
    print('- - - - - -  - -  - - -  - - - -  - - - -  - - - - - -  - -')
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print('- - - - - -  - -  - - -  - - - -  - - - -  - - - - - -  - -')
