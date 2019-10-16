import pandas as pd





if __name__ == '__main__':
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
    weather_test_df = pd.read_csv(f'{dir_loc}/{weather_test_file_name}')
    weather_train_df = pd.read_csv(f'{dir_loc}/{weather_train_file_name}')
    weather_df = pd.concat([weather_train_df, weather_test_df])

    building_metadata_df['floor_count'] = building_metadata_df['floor_count'].fillna(building_metadata_df['floor_count'].median())
    building_metadata_df['year_built'] = building_metadata_df['year_built'].fillna(building_metadata_df['year_built'].mean())
    weather_df['air_temperature'] = weather_df['air_temperature'].fillna(weather_df['air_temperature'].median())
    weather_df['cloud_coverage_present_flag'] = weather_df['cloud_coverage'].apply(lambda x: 0 if pd.isna(x) else 0)
    weather_df['cloud_coverage'] = weather_df['cloud_coverage'].fillna(0)
    weather_df['dew_temperature'] = weather_df['dew_temperature'].fillna(weather_df['dew_temperature'].median())
    weather_df['precip_depth_1_hr_present_flag'] = weather_df['precip_depth_1_hr'].apply(lambda x: 0 if pd.isna(x) else 0)
    weather_df['precip_depth_1_hr'] = weather_df['precip_depth_1_hr'].fillna(weather_df['precip_depth_1_hr'].median())
    weather_df['wind_direction_present_flag'] = weather_df['wind_direction'].apply(lambda x: 0 if pd.isna(x) else 0)
    weather_df['wind_direction'] = weather_df['wind_direction'].fillna(0)
    weather_df['wind_speed'] = weather_df['wind_speed'].fillna(weather_df['wind_speed'].median())

    print(f'train_df.shape: {train_df.shape}')
    train_df = train_df.merge(building_metadata_df)
    print(f'train_df.shape: {train_df.shape}')
    train_df = train_df.merge(weather_df)
    print(f'train_df.shape: {train_df.shape}')


    print(f'test_df.shape: {test_df.shape}')
    test_df = test_df.merge(building_metadata_df)
    print(f'test_df.shape: {test_df.shape}')
    test_df = test_df.merge(weather_df)
    print(f'test_df.shape: {test_df.shape}')




