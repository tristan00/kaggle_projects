import pandas as pd
import lightgbm  as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gc

dir_loc = '/media/td/Samsung_T5/kaggle_data/ASHRAE_2019-10-15/ashrae-energy-prediction'
building_metadata_file_name = 'building_metadata.csv'
sample_submission_file_name = 'sample_submission.csv'
test_file_name = 'test.csv'
train_file_name = 'train.csv'
weather_test_file_name = 'weather_test.csv'
weather_train_file_name = 'weather_train.csv'


def get_data():
    test_df = pd.read_csv(f'{dir_loc}/{test_file_name}')
    train_df = pd.read_csv(f'{dir_loc}/{train_file_name}')
    train_df['train_test'] = 'train'
    test_df['train_test'] = 'test'

    train_test_df = pd.concat([test_df, train_df])
    del test_df, train_df
    gc.collect()

    building_metadata_df = pd.read_csv(f'{dir_loc}/{building_metadata_file_name}')
    building_metadata_df['floor_count'] = building_metadata_df['floor_count'].fillna(building_metadata_df['floor_count'].median())
    building_metadata_df['year_built'] = building_metadata_df['year_built'].fillna(building_metadata_df['year_built'].mean())
    primary_use_le = LabelEncoder()
    building_metadata_df['primary_use'] = primary_use_le.fit_transform(building_metadata_df['primary_use'])
    train_test_df = train_test_df.merge(building_metadata_df, how='left')
    del building_metadata_df
    gc.collect()

    weather_test_df = pd.read_csv(f'{dir_loc}/{weather_test_file_name}')
    weather_train_df = pd.read_csv(f'{dir_loc}/{weather_train_file_name}')
    weather_df = pd.concat([weather_train_df, weather_test_df])
    del test_df, train_df
    gc.collect()


