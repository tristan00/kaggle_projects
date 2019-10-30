import pandas as pd
import lightgbm  as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


dir_loc = '/media/td/Samsung_T5/kaggle_data/ASHRAE_2019-10-15/ashrae-energy-prediction'
building_metadata_file_name = 'building_metadata.csv'
sample_submission_file_name = 'sample_submission.csv'
test_file_name = 'test.csv'
train_file_name = 'train.csv'
weather_test_file_name = 'weather_test.csv'
weather_train_file_name = 'weather_train.csv'

building_metadata_df = pd.read_csv(f'{dir_loc}/{building_metadata_file_name}')
train_df = pd.read_csv(f'{dir_loc}/{train_file_name}')
test_df = pd.read_csv(f'{dir_loc}/{test_file_name}')
train_df['train_test'] = 'train'
test_df['train_test'] = 'test'
train_test_df = pd.concat([test_df, train_df])
del train_df, test_df

weather_test_df = pd.read_csv(f'{dir_loc}/{weather_test_file_name}')
weather_train_df = pd.read_csv(f'{dir_loc}/{weather_train_file_name}')
weather_df = pd.concat([weather_train_df, weather_test_df])

# nan filling
building_metadata_df['floor_count'] = building_metadata_df['floor_count'].fillna(
    building_metadata_df['floor_count'].median())
building_metadata_df['year_built'] = building_metadata_df['year_built'].fillna(
    building_metadata_df['year_built'].mean())
weather_df['air_temperature'] = weather_df['air_temperature'].fillna(weather_df['air_temperature'].median())
weather_df['cloud_coverage_present_flag'] = weather_df['cloud_coverage'].apply(lambda x: 0 if pd.isna(x) else 0)
weather_df['cloud_coverage'] = weather_df['cloud_coverage'].fillna(0)
weather_df['dew_temperature'] = weather_df['dew_temperature'].fillna(weather_df['dew_temperature'].median())
weather_df['precip_depth_1_hr_present_flag'] = weather_df['precip_depth_1_hr'].apply(
    lambda x: 0 if pd.isna(x) else 0)
weather_df['precip_depth_1_hr'] = weather_df['precip_depth_1_hr'].fillna(weather_df['precip_depth_1_hr'].median())
weather_df['wind_direction_present_flag'] = weather_df['wind_direction'].apply(lambda x: 0 if pd.isna(x) else 0)
weather_df['wind_direction'] = weather_df['wind_direction'].fillna(0)
weather_df['wind_speed'] = weather_df['wind_speed'].fillna(weather_df['wind_speed'].median())

train_df['timestamp_dt'] = pd.to_datetime(train_df['timestamp'])
test_df['timestamp_dt'] = pd.to_datetime(test_df['timestamp'])

train_df['hour_of_day'] = train_df['timestamp_dt'].dt.day
train_df['day_of_week'] = train_df['timestamp_dt'].dt.dayofweek
train_df['month_of_year'] = train_df['timestamp_dt'].dt.month
test_df['hour_of_day'] = test_df['timestamp_dt'].dt.day
test_df['day_of_week'] = test_df['timestamp_dt'].dt.dayofweek
test_df['month_of_year'] = test_df['timestamp_dt'].dt.month

train_df = train_df.merge(building_metadata_df, how='left')
train_df = train_df.merge(weather_df, how='left')
test_df = test_df.merge(building_metadata_df, how='left')
test_df = test_df.merge(weather_df, how='left')

primary_use_le = LabelEncoder()
train_df['primary_use'] = primary_use_le.fit_transform(train_df['primary_use'])
test_df['primary_use'] = primary_use_le.transform(test_df['primary_use'])

lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    "learning_rate": 0.1,
    "max_depth": -1,
    'num_leaves': 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    'bagging_freq': 1,
}
num_boost_round = 1000000

feature_cols = ['meter', 'building_id', 'site_id', 'primary_use', 'square_feet', 'year_built', 'floor_count', 'air_temperature',
                'cloud_coverage',
                'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed',
                'cloud_coverage_present_flag', 'precip_depth_1_hr_present_flag', 'wind_direction_present_flag',
                'hour_of_day', 'day_of_week', 'month_of_year']
target = 'meter_reading'

split_train_df, split_val_df = train_test_split(train_df)

lgtrain = lgb.Dataset(split_train_df[feature_cols], split_train_df[target])
lgvalid = lgb.Dataset(split_val_df[feature_cols], split_val_df[target])

model = lgb.train(lgbm_params, lgtrain,
                 valid_sets=[lgtrain, lgvalid],
                 num_boost_round=num_boost_round,
                 valid_names=['train', 'valid'],
                 early_stopping_rounds=10,
                 verbose_eval=10)

test_df['meter_reading'] = model.predict(test_df[feature_cols], num_iteration=model.best_iteration)
test_df = test_df[['row_id', 'meter_reading']]
test_df.to_csv(f'{dir_loc}/preds.csv')

fi = model.feature_importance(iteration=model.best_iteration, importance_type='gain')
fi_dicts = [{'column': i, 'feature_importance':j} for i, j in zip(feature_cols, fi)]
print(fi_dicts)
