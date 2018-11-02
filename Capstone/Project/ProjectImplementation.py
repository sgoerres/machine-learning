# Import external libraries necessary for this project
import pandas as pd
from sklearn import model_selection
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import os

# import my own methods
import Methods.DataCleanup as M
import Methods.Performance as S
import Methods.OutputTree as O

# protect against some multiprocessing issues in windows
# https://joblib.readthedocs.io/en/latest/parallel.html#old-multiprocessing-backend
if __name__ == '__main__':
    # Load data
    train = pd.read_csv('data/train.csv')

    #specify random state and other settings
    random_state = 36#42
    verbose = False
    export = False

    # specify output dir
    output_dir = "Output/"

    if export:
        # create output dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # start data cleanup
    S._s("cleanup")
    unique_cols = ["ID","y"]
    train = M.cleanupData(train, verbose=verbose, export=export,  output_dir=output_dir)
    print(f"Runtime cleanup [s]: {S._rt('cleanup')}")

    # drop all columns that are unique to a single line e.g. ID and result since we want the prediction to focus on the features rather than completely unique information
    cleared_train = train.drop(unique_cols, axis=1)

    # seperate input (x) from output (y)
    x = np.array(cleared_train)
    y = np.array(train["y"])

    # get infoormation about the data
    if verbose:
        print(f"Shape X after cleanup: {x.shape}")
        print(f"Shape Y after cleanup: {y.shape}")

    # split data into training and testing data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, shuffle=True, random_state=random_state)

    # get infoormation about the data
    if verbose:
        print(f"Shape X_train after split: {X_train.shape}")
        print(f"Shape y_train after split: {y_train.shape}")
        print(f"Shape X_test after split: {X_test.shape}")
        print(f"Shape y_tes#t after split: {y_test.shape}")

    # get number of features
    number_of_features = x.shape[1]

    # try out full fitting first -> every feature might have an impact
    # this might lead to either heavy overfitting or a not so bad result
    S._s("adaboost_full")
    adaboost_full = AdaBoostRegressor(DecisionTreeRegressor(max_depth=number_of_features, random_state=random_state), n_estimators=300, random_state=random_state)
    adaboost_tree = adaboost_full.fit(X_train,y_train)
    print(f"fitting adaboost full [s]: {S._rt('adaboost_full')}")

    # use gridearch for further optimization
    # use range with step size 2 to reduce overall calculation amount
    param_grid = {"base_estimator__max_depth" : range(1, number_of_features, 1),#number_of_features
                    "base_estimator__criterion" : ["mse", "mae"],
                  "n_estimators" : range(1, number_of_features, 1)}#number_of_features

    S._s("gridsearch")
    print("GridSearch:")
    grid_search = GridSearchCV(adaboost_full, param_grid=param_grid, scoring = 'r2', n_jobs=-1, verbose=20) #  score by R2
    grid_fit = grid_search.fit(X_train, y_train)
    grid_search_best = grid_fit.best_estimator_
    print(f"Params: {grid_fit.best_params_}")
    print(f"Score: {grid_fit.best_score_}")

    print(f"performing grid_search [s]: {S._rt('gridsearch')}")

    S._s("adaboost_full_predict")
    y_adaboost_full = adaboost_full.predict(X_test)
    print(f"predicting adaboost full [s]: {S._rt('adaboost_full_predict')}")

    # show y_test and y_pred
    if verbose:
        print("y adaboost full: " + y_adaboost_full)
        print("y_test: " + y_test)
        print(adaboost_tree.get_params())

    # export y_test and y_pred
    if export: pd.DataFrame(y_adaboost_full,y_test).to_csv(output_dir+"adaboost_full.csv")

    # calculate R2 score
    adaboost_full_r2 = r2_score(y_test,y_adaboost_full)
    print(f"R2 score: {adaboost_full_r2}")

    # export even more details about the regressors
    if export:
        S._s("export")
        # export resulting decision trees as dot, png, and txt representation
        O.output_all_regressors(adaboost_full,feature_names=cleared_train.columns.values, output_dir=output_dir)
        O.output_plot(y_test,y_adaboost_full,output_dir)
        print(f"Exporting resulting model [s]: {S._rt('export')}")



