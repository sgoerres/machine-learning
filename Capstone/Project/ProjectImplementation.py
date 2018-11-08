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
import Methods.DataStatistic as D

# protect against some multiprocessing issues in windows
# https://joblib.readthedocs.io/en/latest/parallel.html#old-multiprocessing-backend
if __name__ == '__main__':
    print("Starting...")
    # Load data
    train = pd.read_csv('data/train.csv')

    #specify random state and other settings
    random_state = 36#42
    verbose = True
    export = True

    # specify output dir
    output_dir = "Output/"

    if export:
        # create output dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    unique_cols = ["ID","y"]
    D.analyze_base_data(train, export=export, unique_cols=unique_cols, output_dir=output_dir)

    # start data cleanup
    S._s("cleanup")
    train = M.cleanupData(train, verbose=verbose, export=export, output_dir=output_dir)
    print(f"Runtime cleanup [s]: {S._rt('cleanup')}")

    #exit(0)

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
        print(f"Shape y_test after split: {y_test.shape}")

    # get number of features
    number_of_features = x.shape[1]

    # try out full fitting first -> every feature might have an impact
    # this might lead to either heavy overfitting or a not so bad result
    # however this is not after the basic intention of adaboost (many weak learners)
    S._s("adaboost_full")
    adaboost_full = AdaBoostRegressor(DecisionTreeRegressor(max_depth=number_of_features, random_state=random_state), n_estimators=number_of_features, random_state=random_state)
    adaboost_tree = adaboost_full.fit(X_train,y_train)
    print(f"fitting adaboost full [s]: {S._rt('adaboost_full')}")

    S._s("adaboost_full_predict")
    y_adaboost_full = adaboost_full.predict(X_test)
    print(f"predicting adaboost full [s]: {S._rt('adaboost_full_predict')}")

    # use gridearch for further optimization

    base_estimator_max_depth = 25 #int(round(number_of_features * 0.15)) # take 15 percent of number of features as max depth for decisiontreeregressor
    param_grid = {"base_estimator__max_depth" : range(1, base_estimator_max_depth, 1),#number_of_features
                  "n_estimators" : range(1, number_of_features, 1)}#number_of_features


    #NOTE: GridSearch itself is skipped here because of overall runtime
    #      The best estimator has been established using AWS
    #      Params: {'base_estimator__max_depth': 4, 'n_estimators': 4}
    #      Score: 0.5640054576742541
    #      performing grid_search [s]: 23910.188027

    #S._s("gridsearch")
    #print("GridSearch:")
    #grid_search = GridSearchCV(adaboost_full, param_grid=param_grid, scoring = 'r2', n_jobs=-1, verbose=20) #  score by R2
    #grid_fit = grid_search.fit(X_train, y_train)
    #grid_search_best = grid_fit.best_estimator_
    #print(f"Params: {grid_fit.best_params_}")
    #print(f"Score: {grid_fit.best_score_}")
    #print(f"performing grid_search [s]: {S._rt('gridsearch')}")

    S._s("best_estimator_fit")
    best_estimator = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4, random_state=random_state),
                      n_estimators=4, random_state=random_state)

    best_estimator_tree = best_estimator.fit(X_train, y_train)

    print(f"fitting best estimator [s]: {S._rt('best_estimator_fit')}")

    S._s("predict_best")
    y_best = best_estimator.predict(X_test)

    print(f"predicting best estimator [s]: {S._rt('predict_best')}")
    # show y_test and y_pred
    if verbose:
        print("y adaboost full: ")
        print(y_adaboost_full)
        print("y best:          ")
        print(y_best)
        print("y_test:          ")
        print(y_test)
        print("Adaboost Params: ")
        print(adaboost_tree.get_params())
        print("Best Params:     ")
        print(best_estimator_tree.get_params())

    # export y_test and y_pred
    if export: pd.DataFrame(y_adaboost_full,y_test).to_csv(output_dir+"adaboost_full.csv")
    if export: pd.DataFrame(y_best, y_test).to_csv(output_dir + "best.csv")

    # calculate R2 score
    adaboost_full_r2 = r2_score(y_test, y_adaboost_full)
    best_r2 = r2_score(y_test, y_best)
    print(f"Adaboost full R2 score:  {adaboost_full_r2}")
    print(f"Best Estimator R2 score: {best_r2}")

    # export even more details about the regressors
    if export:
        S._s("export")
        # export resulting decision trees as dot, png, and txt representation
        O.output_all_regressors(adaboost_full, feature_names=cleared_train.columns.values, output_dir=output_dir, nameseed="adaboost_full")
        O.output_plot(y_test,y_adaboost_full, output_dir, "adaboost_full")

        O.output_all_regressors(best_estimator, feature_names=cleared_train.columns.values, output_dir=output_dir, nameseed="best_est")
        O.output_plot(y_test, y_best, output_dir, "best_est")

        O.output_plot_3(y_test, y_adaboost_full, y_best, output_dir, "all")
        print(f"Exporting resulting models [s]: {S._rt('export')}")

        print(f"Runtime total [s]: {S._rt()}")
