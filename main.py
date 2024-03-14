import pandas as pd
from create_dfs import data_eng
from create_model import lgbm_model


X_train, y_train, X_test, y_test = data_eng()

importance_plot, mae, saved_model = lgbm_model(X_train, y_train, X_test, y_test)

# print(importance_plot)
# print(mae)
# print(saved_model)