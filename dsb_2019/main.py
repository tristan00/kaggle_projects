import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.ensemble import RandomForestRegressor



def aggregate(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    out_dict = dict()

    out_dict['num_of_events'] = df.shape[0]
    num_of_event_by_code = dict()
    for i in df['event_code']:
        num_of_event_by_code.setdefault('event_code_count_{}'.format(i), 0)
        num_of_event_by_code['event_code_count_{}'.format(i)] += 1
    out_dict.update(num_of_event_by_code)

    out_dict['most_common_game_type'] = df['type'].value_counts().index[0]
    out_dict['most_common_game_world'] = df['type'].value_counts().index[0]
    out_dict['most_common_game_title'] = df['type'].value_counts().index[0]
    out_dict['game_time'] = max(df['game_time'])

    out_dict['event_code_string'] = 'a'.join([str(i) for i in df['event_code']])

    new_df = df.select_dtypes(include=numerics)

    for i in new_df.columns:
        out_dict['col_median_{}'.format(i)] = new_df[i].median()

    return out_dict


class OHE(OneHotEncoder):
    replacement_value = 'nan_cat'
    def __init__(self, n_values=None, categorical_features=None,
                 categories=None, sparse=True, dtype=np.float64,
                 handle_unknown='error', min_perc = .01, col_name = ''):
        super().__init__(n_values=n_values, categorical_features=categorical_features,
                 categories=categories, sparse=sparse, dtype=dtype,
                 handle_unknown=handle_unknown)
        self.min_perc = min_perc
        self.col_name = col_name
        self.valid_values = []
        self.col_names = []
        self.nan_replacement_value = None

    def fit(self, X, y=None):
        input_series = self.process_input(X)
        super().fit( input_series)
        self.col_names = ['{col_base_name}_{value_name}'.format(col_base_name=self.col_name, value_name = i) for i in self.categories_[0]]

    def transform(self, X):
        input_series = self.process_input(X)
        output = super().transform(input_series)
        return self.process_output(output)

    def process_input(self, s):
        if not self.nan_replacement_value:
            self.nan_replacement_value = s.mode()[0]
        s = s.fillna(s.mode())
        s = s.astype(str)

        if not self.valid_values:
            self.valid_values = [i for  i, j in dict(s.value_counts(normalize = True)).items() if j >= self.min_perc]

        prediction_values_to_replace = [i for i in s.unique() if i not in self.valid_values]
        replace_dict = {i: self.replacement_value for i in prediction_values_to_replace}
        replace_dict.update({i:i for i in self.valid_values})
        s = s.map(replace_dict.get)
        return s.values.reshape(-1, 1)

    def process_output(self, output):
        output_df = pd.DataFrame(data = output.toarray(),
                                columns = self.col_names)
        return output_df


def parse_event_data(df):
    event_data_list = [json.loads(i) for i in train_df['event_data'].tolist()]
    events_df = pd.DataFrame.from_dict(event_data_list)
    events_df.columns = ['events_data_{}'.format(i) for i in events_df.columns]
    df = df.join(events_df)
    df = df.drop('event_data', axis = 1)
    return df

path = r'C:\Users\trist\Downloads\data-science-bowl-2019'
train_df = pd.read_csv(f'{path}/train.csv')
test_df = pd.read_csv(f'{path}/test.csv')
train_df['test_set'] = 0
test_df['test_set'] = 1

main_df = pd.concat([train_df, test_df])
train_labels_df = pd.read_csv(f'{path}/train_labels.csv')
specs_df = pd.read_csv(f'{path}/specs.csv')
sample_submission_df = pd.read_csv(f'{path}/sample_submission.csv')

main_df = parse_event_data(main_df)
main_group_df = main_df.groupby(['game_session', 'installation_id', 'test_set']).apply(aggregate)
main_group_df = main_group_df.reset_index()
main_group_df.columns = ['game_session', 'installation_id', 'test_set', 'json']

main_df_group_json = pd.DataFrame.from_dict(main_group_df['json'].tolist())
main_group_df = main_group_df.join(main_df_group_json)
main_group_df = main_group_df.drop('json', axis = 1)

world_ohe = OHE(col_name = 'world')
type_ohe = OHE(col_name = 'type')
title_ohe = OHE(col_name = 'title')
top_event_code_ohe = OHE(col_name = 'top_event_code')

world_ohe.fit(main_group_df['most_common_game_world'])
type_ohe.fit(main_group_df['most_common_game_type'])
title_ohe.fit(main_group_df['most_common_game_title'])

word_df = world_ohe.transform(main_group_df['most_common_game_world'])
type_df = type_ohe.transform(main_group_df['most_common_game_type'])
title_df = title_ohe.transform(main_group_df['most_common_game_title'])
main_group_df = main_group_df.drop(['most_common_game_world', 'most_common_game_type', 'most_common_game_title'], axis = 1)

cv1 = CountVectorizer(binary = True, ngram_range=(9, 13), analyzer='char', max_features = 100)
event_codes_string_np = cv1.fit_transform(main_group_df['event_code_string']).todense()
event_codes_string_df = pd.DataFrame(data = event_codes_string_np,columns = cv1.vocabulary_, index = main_group_df.index)

main_df = main_df.join(word_df)
main_df = main_df.join(type_df)
main_df = main_df.join(title_df)
main_group_df = main_group_df.join(event_codes_string_df)
main_group_df = main_group_df.drop(['event_code_string'], axis = 1)


train_labels_df_small = train_labels_df[['game_session', 'installation_id', 'accuracy_group']]
train_df = main_group_df.merge(train_labels_df_small)

# train_df = labeled_data_df[labeled_data_df['test_set'] == 0]
x_df = train_df[main_group_df.columns]
x_df = x_df.drop(['game_session', 'installation_id'], axis = 1)
y = train_df['accuracy_group']

x_df = x_df.fillna(0)
rf = RandomForestRegressor()
rf.fit(x_df, y)

results = []
for i, j in zip(x_df.columns, rf.feature_importances_):
    results.append({'col':i,'fi':j})

df_res = pd.DataFrame.from_dict(results)
df_res = df_res.sort_values('fi', ascending=False)
df_res.to_csv(f'{path}/fi.csv', index=False)