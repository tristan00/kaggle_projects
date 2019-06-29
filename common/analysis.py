import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

other_key = 'other_category'
nan_key = 'nan_value'
drop_behavior = 'drop'
target_col_name = 'target_col'
random_seed = 1


def get_model(model_type, **kwargs):
    if model_type == 'RandomForest':
        return RandomForestRegressor(**kwargs)
    elif model_type == 'ExtraTreeRandomForest':
        return ExtraTreesRegressor(**kwargs)
    elif model_type == 'Linear':
        return LinearRegression(**kwargs)
    elif model_type == 'DecisionTree':
        return DecisionTreeRegressor(**kwargs)
    else:
        raise Exception('invalid base_model: {0}'.format(model_type))

class EDA:
    def __init__(self, x, y,
                 columns_to_ohe = (),
                 columns_to_label_encode = (),
                 problem_type = 'regression',
                 max_one_hot_encoding_categories = 12,
                 min_data_in_one_hot_cat = 0.0,
                 invalid_column_behavior = 'drop',
                 invalid_columns = (),
                 base_models = ('RandomForest', 'ExtraTreeRandomForest', 'Linear', 'DecisionTree')):
        '''
        :param x: Dataframe without label
        :param y: labels as a series
        :param columns_to_ohe: Columns to one hot encode
        :param columns_to_label_encode: columns to label encode, Not implemented
        :param label_type: regression supported, will add classification later
        :param invalid_column_behavior: drop to drop the column, error to throw error
        :param default_model: RandomForest, ExtraTreeRandomForest, Linear and DecisionTree are supported
        '''

        self.x = x
        self.y = y
        self.processed_x = pd.DataFrame()
        self.columns_to_ohe = columns_to_ohe
        self.columns_to_label_encode = columns_to_label_encode
        self.problem_type = problem_type
        self.max_one_hot_encoding_categories = max_one_hot_encoding_categories
        self.min_data_in_one_hot_cat = min_data_in_one_hot_cat
        self.invalid_column_behavior = invalid_column_behavior
        self.invalid_columns = invalid_columns
        self.base_models = base_models

        self.encoders = dict()
        self.split_columns = dict()
        self.other_column_mapping = dict()
        self.columns_type = dict()
        self.linear_df = pd.DataFrame()
        self.tree_df = pd.DataFrame()
        self.main_df = pd.DataFrame()
        self.make_data_valid()

    def make_data_valid(self):
        invalid_cols = self.invalid_columns
        if other_key in self.x.columns:
            invalid_cols.append(other_key)
        invalid_cols.extend(self.pre_analyze_data())
        if invalid_cols:
            if self.invalid_column_behavior == drop_behavior:
                for i in invalid_cols:
                    self.x = self.x.drop(i, axis = 1)
            else:
                raise Exception('Invalid columns: {0}'.format(invalid_cols))

    def pre_analyze_data(self):
        invalid_cols = []
        for col in self.x.columns:
            if col in self.columns_to_ohe or self.x[col].dtype != np.number:
                self.columns_type[col] = 0
            elif col in self.columns_to_label_encode:
                self.columns_type[col] = 1
            elif self.x[col].dtype == np.number:
                self.columns_type[col] = 2
            else:
                invalid_cols.append(col)
        return invalid_cols

    def prepare_data(self):
        self.processed_x_list = []

        for col in self.x.columns:
            if self.columns_type[col] == 0:
                self.encoders[col] = OneHotEncoder()
                col_copy = self.x[col].copy().fillna(nan_key).astype(str)
                col_copy_value_counts = dict(col_copy.value_counts())
                sorted_categories = sorted([(i, j) for i, j in col_copy_value_counts.items()], key = lambda x: x[1], reverse=True)
                if self.max_one_hot_encoding_categories:
                    sorted_categories = sorted_categories[:self.max_one_hot_encoding_categories]
                if self.min_data_in_one_hot_cat:
                    sorted_categories = [i for i in sorted_categories if i[1]/self.x.shape[0] >= self.min_data_in_one_hot_cat]
                sorted_categories = [i[0] for i in sorted_categories]
                invalid_categories = [i for i in col_copy_value_counts if i not in sorted_categories]
                self.other_column_mapping[col] = invalid_categories
                for i in self.other_column_mapping[col]:
                    col_copy = col_copy.replace(i, other_key)
                self.split_columns[col] = []
                self.encoders[col].fit(col_copy.values.reshape(-1, 1))
                for i in self.encoders[col].categories_[0]:
                    self.split_columns[col].append(str(col) + '_' + str(i))
                col_representation = self.encoders[col].transform(col_copy.values.reshape(-1, 1)).toarray()

                col_representation_df = pd.DataFrame(columns = self.split_columns[col],
                                                     data = col_representation)
                self.processed_x_list.append(col_representation_df)
            elif self.columns_type[col] == 1:
                raise NotImplemented('label encoding not implemented')
            elif self.columns_type[col] == 2:
                self.processed_x_list.append(self.x[col].fillna(self.x[col].median()).to_frame())
                self.split_columns[col] = [col]
        self.processed_x = pd.concat(self.processed_x_list, axis = 1)

    def calculate_linear_feature_importance(self):
        linear_data = []

        for col, subs in self.split_columns.items():
            for s in subs:
                slope, intercept, r_value, p_value, std_err = stats.linregress(self.processed_x[s].values, self.y)
                linear_data.append({'slope': slope,
                                    'intercept': intercept,
                                    'r_value': r_value,
                                    'r2_value': r_value*r_value,
                                    'p_value': p_value,
                                    'std_err': std_err,
                                    'original_column': col,
                                    'column_name': s})
        self.linear_df = pd.DataFrame.from_dict(linear_data)
        self.linear_df = self.linear_df.sort_values('p_value')

    def calculate_tree_feature_importance(self):
        tree_data = []
        original_features_sum = dict()
        model = RandomForestRegressor()
        model.fit(self.processed_x, self.y)

        for f_name, f_entropy_reduction in zip(self.processed_x.columns, model.feature_importances_):
            for col, subs in self.split_columns.items():
                for s in subs:
                    if s == f_name:
                        tree_data.append({'column_name': f_name,
                                          'original_column': col,
                                          'random_forest_entropy_reduction':f_entropy_reduction})
                        original_features_sum.setdefault(col, 0)
                        original_features_sum[col] += f_entropy_reduction

        for i in tree_data:
            i.update({'original_features_entropy_reduction_sum': original_features_sum[i['original_column']]})
        self.tree_df = pd.DataFrame.from_dict(tree_data)

    def calculate_feature_importance_alone(self):
        feature_alone_score = []

        for col, subs in self.split_columns.items():
            col_rec = {'original_column': col}
            x_sub = self.processed_x[subs]

            for m in self.base_models:
                model = get_model(m)
                score = cross_val_score(model, x_sub, self.y, cv = 3)
                col_rec.update({str(m) + '_single_column_score': sum(score)/len(score)})
            feature_alone_score.append(col_rec)

        self.col_alone_df = pd.DataFrame.from_dict(feature_alone_score)

    def calculate_feature_importance_removal(self):
        '''
        Concept of SHAP values
        '''
        feature_alone_score = []

        for col, subs in self.split_columns.items():
            col_rec = {'original_column': col}
            x_copy = self.processed_x.copy()
            for i in subs:
                x_copy = x_copy.drop(i, axis = 1)

            for m in self.base_models:
                model = get_model(m)
                score = cross_val_score(model, x_copy, self.y, cv=3)
                col_rec.update({str(m) + '_col_removed_score': sum(score)/len(score)})
            feature_alone_score.append(col_rec)

        self.col_removed_df = pd.DataFrame.from_dict(feature_alone_score)

    def calculate_best_feature_in_combination(self):
        pass

    def per_feature_analysis(self):
        combined_df = self.x.copy()
        combined_df[target_col_name] = self.y.copy()

        for col in self.x.columns:
            if self.columns_type[col] == 0:
                res = combined_df.grouby(col)[target_col_name]

    def combine_feature_importances(self):
        self.main_df = self.linear_df.merge(self.tree_df)
        self.original_col_dfs = self.col_removed_df.merge(self.col_alone_df)

    def run_analysis(self):
        self.prepare_data()
        self.calculate_linear_feature_importance()
        self.calculate_tree_feature_importance()
        self.calculate_feature_importance_removal()
        self.calculate_feature_importance_alone()
        self.combine_feature_importances()
        return self.main_df, self.original_col_dfs


if __name__ == '__main__':
    path = r'C:\Users\trist\OneDrive\Desktop\house-prices-advanced-regression-techniques'
    train_data = pd.read_csv('{0}/train.csv'.format(path))
    train_x = train_data.drop('SalePrice', axis = 1)
    train_y = train_data['SalePrice']
    eda = EDA(train_x, train_y, invalid_columns=['Id'])
    df1, df2 = eda.run_analysis()
    df1.to_csv('{0}/df1.csv'.format(path))
    df2.to_csv('{0}/df2.csv'.format(path))

