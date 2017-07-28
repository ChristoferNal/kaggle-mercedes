from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas as pd
import random
import lightgbm as lgb


print('Loading data ...')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


train = pd.get_dummies(train, drop_first=True)
y_train = train['y']

train.drop(['ID','y'],axis=1,inplace=True)

print ('Number of features : %d' % train.shape[1])

all_features = train.columns

# number of iterations is the only stopping criteria
##TO DO: implement early stopping based on not getting better score after some iterations

NUM_OF_ITER = 100

solutions_dict = dict()
solutions_dict_new = dict()
NUM_SOLUTIONS =60
NUM_FEATURES = train.shape[1]
print(NUM_FEATURES)


NUMBER_OF_SELECTED_SOLUTIONS = NUM_SOLUTIONS/2

#----------------Create initial Random Solutions ------------------------------
#we create randomly the initial population
for i in range(NUM_SOLUTIONS):
    prop = random.uniform(0.05,1)
    solutions_dict[i] = np.random.choice([0, 1], size=(NUM_FEATURES), p=[prop, 1-prop])


for i in range(NUM_OF_ITER):
    print('numner od iter-------' + str(i))
    solutions_scores = np.zeros(NUM_SOLUTIONS)
    #------------- Scoring Each Solution-------------------------------
    #we use as Fit Functions the mean value of a 5-Fold cross validation  based on Lightboost Algo
    for k in range(NUM_SOLUTIONS):
        cols = []
        solution = solutions_dict[k]
        for j in range(NUM_FEATURES):
            if solution[j]==1:
                cols.append(all_features[j])
        # print('--------cols_-----'+ str(len(cols)))

        et = lgb.LGBMRegressor(boosting_type='gbdt', subsample=1, subsample_freq=1,n_estimators=50, max_depth=3,scale_pos_weight=5,is_unbalance = True,
                                                seed=0, nthread=-1, silent=True, reg_alpha = 0, reg_lambda=1)

        results = cross_val_score(et, train[cols].values, y_train, cv=5, scoring='r2')
        # print('--------cols_-----' + str(len(cols)))
        # print("LGBM score: %.4f (%.4f)" % (results.mean(), results.std()))
        solutions_scores[k]=results.mean()
    print(solutions_scores.max())


    #-----------------------Selection------------------------------
    ##selecting halh solution with the biggest score (it can be more

    ind = np.argpartition(solutions_scores, -NUMBER_OF_SELECTED_SOLUTIONS)[-NUMBER_OF_SELECTED_SOLUTIONS:]
    # print(ind)
    solutions_dict_new = solutions_dict
    for a in range(NUMBER_OF_SELECTED_SOLUTIONS):
        solutions_dict[a] = solutions_dict_new[ind[a]]

    #--------------------------Crossover------------------------------
    #Only one crossing - point implemented here

    for b in range(NUMBER_OF_SELECTED_SOLUTIONS, NUM_SOLUTIONS,2):
        proportion = random.randint(0, NUM_FEATURES - 1)
        solutions_dict[b] = np.concatenate((solutions_dict[b-NUMBER_OF_SELECTED_SOLUTIONS][:proportion],solutions_dict[b-NUMBER_OF_SELECTED_SOLUTIONS+1][proportion:]),axis=0)
        solutions_dict[b+1] = np.concatenate((solutions_dict[b-NUMBER_OF_SELECTED_SOLUTIONS+1][:proportion], solutions_dict[b-NUMBER_OF_SELECTED_SOLUTIONS][proportion:]), axis=0)
    # print(len(solutions_dict))

    # -----------------------Mutation--------------------------------
    MUTATION_THRESHOLD = 0.15

    for c in range(NUM_SOLUTIONS):
        mutation_random = random.uniform(0,1)
        mutation_index = random.randint(0, NUM_FEATURES - 1)
        if mutation_random < MUTATION_THRESHOLD:
            solutions_dict[c][mutation_index] = 1-solutions_dict[c][mutation_index]

#getting the best solution by sorting them
ind = np.argpartition(solutions_scores, -1)[-1:]
solutions_dict_new = solutions_dict
for a in range(1):
    solutions_dict[a] = solutions_dict_new[ind[a]]

#getting the columns in the best solution
for k in range(1):
    cols = []
    solution = solutions_dict[k]
    for j in range(NUM_FEATURES):
        if solution[j]==1:
            cols.append(all_features[j])


print(cols)

