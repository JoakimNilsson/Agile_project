import pandas as pd

def data_eng():

    print('Running Data Cleaning and Engeneering')
    import pandas as pd

    print('Creating Dataframes')
    train_data = pd.read_csv(r'C:\Users\jocke\Desktop\Skola\Agilt\optiver-trading-at-the-close\train.csv')
    test_data = pd.read_csv(r'C:\Users\jocke\Desktop\Skola\Agilt\optiver-trading-at-the-close\example_test_files\test.csv')
    reavealed_targets = pd.read_csv(r'C:\Users\jocke\Desktop\Skola\Agilt\optiver-trading-at-the-close\example_test_files\revealed_targets.csv')
    sample_submission = pd.read_csv(r'C:\Users\jocke\Desktop\Skola\Agilt\optiver-trading-at-the-close\example_test_files\sample_submission.csv')

    print('Cleaning the data')
    reavealed_targets = reavealed_targets.dropna()
    reavealed_targets['date_id'] = reavealed_targets['date_id'].astype(int)
    reavealed_targets['seconds_in_bucket'] = reavealed_targets['seconds_in_bucket'].astype(int)
    reavealed_targets['stock_id'] = reavealed_targets['stock_id'].astype(int)

    reavealed_targets['row_id'] = reavealed_targets['date_id'].astype(str) + '_' + reavealed_targets['seconds_in_bucket'].astype(str) + '_' +  reavealed_targets['stock_id'].astype(str)

    merged_df = pd.merge(test_data, reavealed_targets, on='row_id', how='inner')

    merged_df = merged_df.drop(['stock_id_y', 'date_id_y', 'seconds_in_bucket_y', 'time_id_y', 'revealed_date_id', 'revealed_time_id'], axis=1)

    merged_df.rename(columns={'seconds_in_bucket_x': 'seconds_in_bucket'}, inplace=True)
    merged_df.rename(columns={'stock_id_x': 'stock_id'}, inplace=True)
    merged_df.rename(columns={'date_id_x': 'date_id'}, inplace=True)
    merged_df.rename(columns={'time_id_x': 'time_id'}, inplace=True)

    train_data.fillna(0, inplace=True)
    merged_df.fillna(0, inplace=True)

    train_data = train_data.dropna()

    print('Creating Train and test sets')
    # X_train = train_data.drop(['stock_id', 'date_id', 'time_id', 'target', 'row_id'], axis=1)
    X_train = train_data.drop(['target', 'row_id'], axis=1)
    y_train = train_data['target']
    # X_test = merged_df.drop(['stock_id_x', 'date_id_x', 'time_id_x', 'row_id', 'revealed_target'], axis=1)
    X_test = merged_df.drop(['revealed_target', 'row_id'], axis=1)
    y_test = merged_df['revealed_target']

    X_train = pd.get_dummies(X_train, columns=['imbalance_buy_sell_flag'], drop_first=False)
    X_test = pd.get_dummies(X_test, columns=['imbalance_buy_sell_flag'], drop_first=False)

    X_train['imbalance_ratio'] = X_train['imbalance_size'] / X_train['matched_size'] 
    X_train['imbl_size1'] = (X_train['bid_size']-X_train['ask_size']) / (X_train['bid_size']+X_train['ask_size'])
    X_train['imbl_size2'] = (X_train['imbalance_size']-X_train['matched_size']) / (X_train['imbalance_size']+X_train['matched_size'])

    X_test['imbalance_ratio'] = X_test['imbalance_size'] / X_test['matched_size'] 
    X_test['imbl_size1'] = (X_test['bid_size']-X_test['ask_size']) / (X_test['bid_size']+X_test['ask_size'])
    X_test['imbl_size2'] = (X_test['imbalance_size']-X_test['matched_size']) / (X_test['imbalance_size']+X_test['matched_size'])
    
    print('Done with the data process!')

    return X_train, y_train, X_test, y_test