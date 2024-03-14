import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

def lgbm_model(X_train, y_train, X_test, y_test):

    import lightgbm as lgbm
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint as sp_randint
    import joblib
    import matplotlib.pyplot as plt
    import json
    from sklearn.metrics import mean_absolute_error
    import pickle
    import os

    print('Loading in the model')
    lgbm1 = lgbm.LGBMRegressor()

    print('Creating param_dist')
    param_dist = {
    'n_estimators': (900, 1000, 1100),
    'max_depth': (7, 25, 37),
    'num_leaves': (35, 40, 50, 70),
    'learning_rate': [0.001],
    'subsample': [0.8, 0.9, 1.0],
    'reg_alpha': [0],
    'reg_lambda': [0.5, 0.7, 1],
    'force_row_wise': [True]
    }

    print('Starting Random Search')
    random_search = RandomizedSearchCV(lgbm1, param_distributions=param_dist, n_iter=25, cv=3, verbose=2, n_jobs=1)

    print('Fitting Random Search')
    random_search.fit(X_train, y_train)


    print('Predicting on test data')
    y_pred = random_search.predict(X_test)

    print('Getting the MEA for the model!')

    mae = mean_absolute_error(y_test, y_pred)

    print(mae)

    mae_underscore = str(mae).replace('.','_')

    try:
        folder_path = './trails'

        os.mkdir(folder_path)
    except:
         pass

    folder_path = f'./trails/{mae_underscore}'

    os.mkdir(folder_path)

    print('Saving Plot importance')
    # lgbm.plot_importance(random_search,importance_type="gain", max_num_features=10, figsize=(10,6))
    # plt.savefig('importance_plot.png')
    feature_importance = random_search.best_estimator_.feature_importances_
    feature_names = X_train.columns.tolist()

    with open(f'./trails/{mae_underscore}/feature_importance.txt', 'w') as file:
            file.write(f'Feature importace for model with score: {mae}' + '\n')
    
    for i in range(len(feature_names)):
        with open(f'./trails/{mae_underscore}/feature_importance.txt', 'a') as file:
            file.write(f'{feature_names[i]}: {feature_importance[i]}' + '\n')

    print('Get the best params')
    best_params = random_search.best_params_
    print(best_params)
    with open(f'./trails/{mae_underscore}/best_hyperparameters.json', 'w') as file:
        json.dump(best_params, file)

    with open(f'./trails/{mae_underscore}/best_model.pkl', 'wb') as file:
        joblib.dump(random_search, file)


    return